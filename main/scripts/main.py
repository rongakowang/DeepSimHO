import os
import random
from time import time
from typing import Optional, Union
import sys
sys.path.append('.')
import numpy as np
import torch
from anakin.criterions.criterion import Criterion
from anakin.datasets.hodata import ho_collate
from anakin.metrics.evaluator import Evaluator
from anakin.models.arch import Arch
from anakin.opt import arg, cfg
from anakin.utils import builder
from anakin.utils.etqdm import etqdm
from anakin.utils.logger import logger
from anakin.utils.misc import CONST, TrainMode
from anakin.utils.netutils import build_optimizer, build_scheduler
from anakin.utils.recorder import Recorder
from anakin.utils.summarizer import Summarizer
from anakin.criterions.stabilityloss import MLPMeshMixRegPretrainLoss
from termcolor import colored
torch.autograd.set_detect_anomaly(True)

os.system('export MUJOCO_GL=EGL')
os.system('export PYOPENGL_PLATFORM=egl')

def _init_fn(worker_id):
    seed = int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)

def setup_seed(seed, conv_repeatable=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["MUJOCO_GL"] = "EGL"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_repeatable:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.warning("Exp result NOT repeatable!")


def epoch_pass(
    train_mode: TrainMode,
    epoch_idx: int,
    data_loader: Optional[torch.utils.data.DataLoader],
    arch_model: Union[Arch, torch.nn.DataParallel],
    optimizer_generator: Optional[torch.optim.Optimizer],
    optimizer_discriminator: Optional[torch.optim.Optimizer],
    optimizer_fine_tune: Optional[torch.optim.Optimizer],
    criterion: Optional[Criterion],
    evaluator: Optional[Evaluator],
    summarizer: Optional[Summarizer],
    grad_clip: Optional[float] = None,
    loss_avg: Optional[dict] = None,
):
    if train_mode == TrainMode.TRAIN:
        arch_model.train()
    else:
        arch_model.eval()

    logger.info(f"Model total parameters: {sum(p.numel() for p in list(arch_model.parameters()))/ 1e6}")

    if evaluator:
        evaluator.reset_all()

    bar = etqdm(data_loader)
    if loss_avg:
        bar2 = etqdm(data_loader)
    for batch_idx, batch in enumerate(bar):
        batch['sample_id'] = (train_mode, epoch_idx, batch_idx)
        predict_arch_dict = arch_model(batch)
        
        predicts = {}
        for key in predict_arch_dict.keys():
            predicts.update(predict_arch_dict[key])

        # ==== criterion >>>>
        if criterion:
            final_loss, losses = criterion.compute_losses(predicts, batch)
        else:
            final_loss, losses = torch.Tensor([0.0]), {}

        # <<<<<<<<<<<<<<<<<<<<

        # >>>> evaluate >>>>
        if train_mode == TrainMode.TRAIN:
            if evaluator and batch_idx % 10 == 0:
                evaluator.feed_all(predicts, batch, losses)
        else:
            if evaluator:
                evaluator.feed_all(predicts, batch, losses)
        # <<<<<<<<<<<<<<<<<<

        # >>>> summarize >>>>
        if summarizer is not None and train_mode == TrainMode.TRAIN:
            summarizer.summarize_losses(losses)
        # <<<<<<<<<<<<<<<<<<

        # >>>> backward >>>>
        if train_mode == TrainMode.TRAIN:
            optimizer_generator.zero_grad()
            optimizer_fine_tune.zero_grad()
            optimizer_discriminator.zero_grad()

            final_loss.backward() # has stability, but no discriminator loss
            torch.nn.utils.clip_grad_norm_(arch_model.parameters(), grad_clip) # clip_grad larger for the stability
            optimizer_generator.step() # generator step at here
            
            # clean all gradient
            optimizer_generator.zero_grad()  # for safety
            optimizer_fine_tune.zero_grad()
            optimizer_discriminator.zero_grad()

            predict_arch_dict2 = arch_model(batch)

            predicts2 = {}
            for key in predict_arch_dict2.keys():
                predicts2.update(predict_arch_dict2[key])

            _, losses_stb = MLPMeshMixRegPretrainLoss()(predicts2, batch)

            losses_stb['stability_loss'].backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(arch_model.parameters(), grad_clip) # clip_grad larger for the stability
            optimizer_fine_tune.step()

            optimizer_generator.zero_grad()  # for safety
            optimizer_fine_tune.zero_grad()
            optimizer_discriminator.zero_grad()

            if 'discriminator_loss' in losses_stb.keys():
                losses_stb['discriminator_loss'].backward()
                torch.nn.utils.clip_grad_norm_(arch_model.parameters(), grad_clip)
                optimizer_discriminator.step() # comment out to free it

            # sanity clean
            optimizer_generator.zero_grad()  # for safety
            optimizer_fine_tune.zero_grad()
            optimizer_discriminator.zero_grad()
        # <<<<<<<<<<<<<<<<<<

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<
        bar_perfixes = {
            TrainMode.TRAIN: colored("train", "white", attrs=["bold"]),
            TrainMode.VAL: colored("val", "yellow", attrs=["bold"]),
            TrainMode.TEST: colored("test", "magenta", attrs=["bold"]),
        }
        if loss_avg:
            # extract interested loss
            loss_output = {k: v.item() for k, v in losses.items() if k in loss_avg.keys()} 
            loss_output.update({k: v.item() for k, v in losses_stb.items() if k in loss_avg.keys()})
            for k in loss_avg.keys():
                try:
                    loss_avg[k].append(loss_output[k])
                except:
                    loss_avg[k].append(torch.tensor(0.))

        if loss_avg:
            bar2.set_description(f"{bar_perfixes[train_mode]} loss | {' '.join([f'{k}:{sum(v) / len(v):.5f}' for k,v in loss_avg.items()])}")
                
        bar.set_description(f"{bar_perfixes[train_mode]} Epoch {epoch_idx} | {str(evaluator)}")


def main_worker(time_f: float):
    recorder = Recorder(arg.exp_id, cfg, time_f=time_f)
    summarizer = Summarizer(arg.exp_id, cfg, time_f=time_f)
    logger.info(f"dump args: {arg, cfg['TRAIN']}")

    # region >>>>>>>>>>>>>>>>>>>> load test data >>>>>>>>>>>>>>>>>>>>
    test_data = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"])
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=int(arg.workers),
        drop_last=False,
        collate_fn=ho_collate,
        worker_init_fn=_init_fn,
    )
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # region >>>>>>>>>>>>>>>>>>>> load model >>>>>>>>>>>>>>>>>>>>
    model_list = builder.build_arch_model_list(cfg["ARCH"], preset_cfg=cfg["DATA_PRESET"])
    model = Arch(cfg, model_list=model_list)

    recorder.record_arch_graph(model)
    model = torch.nn.DataParallel(model).to(arg.device)

    generator_params = [p for name, p in model.module.named_parameters() if 'discriminator' not in name]
    fine_tune_params = [p for name, p in model.module.named_parameters() if 'box_head' in name or 'hybrid_head' in name]
    discriminator_params = [p for name, p in model.module.named_parameters() if 'discriminator' in name]

    optimizer_generator = build_optimizer(
        generator_params,
        **cfg["TRAIN"],
    )

    optimizer_fine_tune = build_optimizer(
        fine_tune_params,
        **cfg["TRAIN"],
    )

    optimizer_discriminator = build_optimizer(
        discriminator_params,
        **cfg["TRAIN"],
    )

    scheduler_generator = build_scheduler(optimizer_generator, **cfg["TRAIN"])
    scheduler_discriminator = build_scheduler(optimizer_discriminator, **cfg["TRAIN"])
    scheduler_fine_tune = build_scheduler(optimizer_fine_tune, **cfg["TRAIN"])

    grad_clip = cfg["TRAIN"].get("GRAD_CLIP")
    if grad_clip is not None:
        logger.warning(f"Use gard clip norm {grad_clip}")

    if arg.resume: # use pretrain instead
        epoch = recorder.resume_checkpoints(model, optimizer_generator, scheduler_generator, arg.resume)
        if arg.evaluate:
            cfg["TRAIN"]["EPOCH"] = epoch + 1  # enter into the train loop
    else:
        epoch = 0
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # region >>>>>>>>>>>>>>>>>>>> load criterion >>>>>>>>>>>>>>>>>>>>
    loss_list = builder.build_criterion_loss_list(cfg["CRITERION"],
                                                  preset_cfg=cfg["DATA_PRESET"],
                                                  LAMBDAS=cfg["LAMBDAS"])

    criterion = Criterion(cfg, loss_list=loss_list)
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # region >>>>>>>>>>>>>>>>>>>> load evaluator >>>>>>>>>>>>>>>>>>>>
    metrics_list = builder.build_evaluator_metric_list(cfg["EVALUATOR"], preset_cfg=cfg["DATA_PRESET"])
    evaluator = Evaluator(cfg, metrics_list=metrics_list)
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # region >>>>>>>>>>>>>>>>>>>> load training >>>>>>>>>>>>>>>>>>>>
    train_data = builder.build_dataset(cfg["DATASET"]["TRAIN"], preset_cfg=cfg["DATA_PRESET"])
    logger.info(f"Total training data number is {len(train_data)}")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=int(arg.workers),
        drop_last=False,
        collate_fn=ho_collate,
        worker_init_fn=_init_fn,
    )
    # endregion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Preprare train or testf
    n_epoches = cfg["TRAIN"]["EPOCH"]

    # >>>>>>>>>>>>>>>>>>>> train >>>>>>>>>>>>>>>>>>>>
    logger.info(f"start training from {epoch} to {n_epoches}")
    for epoch_idx in range(epoch, n_epoches):

        if not arg.evaluate:

            loss_terms = ['joints_3d_loss', 'sym_corners_3d_loss',
                           'stability_loss', 'discriminator_loss']

            loss_avg = {}
            for k in loss_terms:
                loss_avg[k] = []
            epoch_pass(
                train_mode=TrainMode.TRAIN,
                epoch_idx=epoch_idx,
                data_loader=train_loader,
                arch_model=model,
                optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator,
                optimizer_fine_tune=optimizer_fine_tune,
                criterion=criterion,
                evaluator=evaluator,
                summarizer=summarizer,
                grad_clip=grad_clip,
                loss_avg=loss_avg
            )
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_fine_tune.step()
            # >>>>>>>>>>>>>>>>>>>> Save checkpoint >>>>>>>>>>>>>>>>>>>>
            recorder.record_checkpoints(model, optimizer_generator, scheduler_generator, epoch_idx, 1) # save every epoch
            recorder.record_evaluator(evaluator, epoch_idx, TrainMode.TRAIN)
            summarizer.summarize_evaluator(evaluator, epoch_idx, train_mode=TrainMode.TRAIN)
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if True: # evaluate at every epoch
            with torch.no_grad():
                epoch_pass(
                    train_mode=TrainMode.TEST,
                    epoch_idx=epoch_idx,
                    data_loader=test_loader,
                    arch_model=model,
                    optimizer_generator=None,
                    optimizer_discriminator=None,
                    optimizer_fine_tune=None,
                    criterion=criterion,
                    evaluator=evaluator,
                    summarizer=summarizer,
                )
            recorder.record_evaluator(evaluator, epoch=epoch_idx, train_mode=TrainMode.TEST)
            summarizer.summarize_evaluator(evaluator, epoch_idx, train_mode=TrainMode.TEST)
            if arg.evaluate: # one epoch only
                break

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def main():
    exp_time = time()
    setup_seed(cfg["TRAIN"]["MANUAL_SEED"], cfg["TRAIN"].get("CONV_REPEATABLE", True))
    logger.info("====> Use Data Parallel <====")
    main_worker(exp_time)  # need to pass in renderer process group info


if __name__ == "__main__":
    main()
