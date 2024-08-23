"""
Pytorch Lightning training interface.

This module is somewhat experimental. Using pytorch lightning
successfully in a distributed context may require understanding
and adjusting the various settings related to parallelism, e.g.
multiprocessing context, torch ddp backend, and how they interact
with your HPC environment.

Some features of hippynn experiments may not be implemented yet.
"""
import torch

import pytorch_lightning as pl
from .routines import TrainingModules
from ..databases import Database
from .routines import SetupParams, setup_training
from ..graphs import GraphModule
from .controllers import Controller
from .metric_tracker import MetricTracker
from .step_functions import get_step_function, StandardStep
from ..plotting import PlotMaker
from ..tools import print_lr


class HippynnLightningModule(pl.LightningModule):
    def __init__(self,model: GraphModule,
                 loss: GraphModule,
                 eval_loss: GraphModule,
                 eval_names: list[str],
                 stopping_key: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 controller: Controller,
                 metric_tracker: MetricTracker,
                 plot_maker: PlotMaker,
                 inputs:list[str],
                 targets:list[str],
                 n_outputs:int,
                 *args,**kwargs): # forwards args and kwargs to where?
        super().__init__()

        self.model = model
        self.loss = loss
        self.eval_loss = eval_loss
        self.eval_names = eval_names
        self.stopping_key = stopping_key
        self.controller = controller
        self.metric_tracker = metric_tracker
        self.optimizer = optimizer # does this conflict with PL names?
        self.scheduler = scheduler # does this conflict with PL names?
        self.inputs = inputs
        self.targets = targets
        self.n_inputs = len(self.inputs)
        self.n_targets = len(self.targets)
        self.n_outputs = n_outputs
        self.plot_maker = plot_maker

        # Storage for predictions across batches for eval mode.
        self.eval_step_outputs = []

        if not isinstance(step_fn:=get_step_function(optimizer), StandardStep):  # :=
            raise NotImplementedError(f"Optimzers with non-standard steps are not yet supported. {optimizer,step_fn}")
        if args or kwargs:
            raise NotImplementedError("No args or kwargs support yet.")

    @classmethod
    def from_experiment_setup(cls, training_modules: TrainingModules, database: Database, setup_params:SetupParams, **kwargs):
        training_modules, controller, metric_tracker = setup_training(training_modules, setup_params)
        return cls.from_train_setup(training_modules, database, controller, metric_tracker, **kwargs)

    @classmethod
    def from_train_setup(cls,
                         training_modules: TrainingModules,
                         database: Database,
                         controller: Controller,
                         metric_tracker: MetricTracker,
                         callbacks=None,
                         batch_callbacks=None,
                         **kwargs,
                         ):

        model, loss, evaluator = training_modules

        eval_names = evaluator.loss_names
        trainer = cls(
            model = model,
            loss = loss,
            eval_loss = evaluator.loss,
            eval_names = evaluator.loss_names,
            optimizer = controller.optimizer,
            scheduler = controller.scheduler,
            stopping_key = controller.stopping_key,
            controller = controller,
            metric_tracker = metric_tracker,
            plot_maker = evaluator.plot_maker,
            inputs = database.inputs,
            targets = database.targets,
            n_outputs =  evaluator.n_outputs,
            **kwargs,
        )

        # pytorch lightning is now in charge of stepping the scheduler.
        controller.scheduler_list = []

        if callbacks is not None or batch_callbacks is not None:
            return NotImplemented("arbitrary callbacks are not yet supported with pytorch lightning.")

        return trainer, HippynnDataModule(database, controller.batch_size)

    def configure_optimizers(self):

        config = dict(optimizer=self.optimizer)

        if self.scheduler is not None:
            config["lr_scheduler"] = {
                "scheduler": self.scheduler,
                "interval": "epoch",  # can be epoch or step
                "frequency": 1,# How many intervals should pass between calls to  `scheduler.step()`.
                "monitor": "valid_" + self.stopping_key, # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "strict": True,
                "name": "learning_rate",
            }

        return config

    def on_train_epoch_start(self):
        print_lr(self.optimizer,print_=self.print)


    def training_step(self, batch, batch_idx):

        batch_inputs = batch[:self.n_inputs]
        batch_targets = batch[-self.n_targets:]

        batch_model_outputs = self.model(*batch_inputs)
        batch_train_loss = self.loss(*batch_model_outputs, *batch_targets)[0]

        self.log("train_loss", batch_train_loss)
        return batch_train_loss

    def _eval_step(self, batch, batch_idx):


        batch_inputs = batch[: self.n_inputs]
        batch_targets = batch[-self.n_targets:]

        batch_dict = dict(zip(self.inputs, batch_inputs))

        # it is very very common to fit to derivatives, e.g. force, in hippynn. Override lightning default.
        with torch.autograd.set_grad_enabled(True):
            batch_predictions = self.model(*batch_inputs)

        batch_predictions = [bp.detach() for bp in batch_predictions]

        outputs = (batch_predictions, batch_targets)
        self.eval_step_outputs.append(outputs)
        return batch_predictions

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch,batch_idx)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch,batch_idx)


    def _eval_epoch_end(self, prefix):

        all_batch_predictions, all_batch_targets = zip(*self.eval_step_outputs)
        # now 'shape' (n_batch, n_outputs) -> need to transpose.
        all_batch_predictions = [[bpred[i] for bpred in all_batch_predictions] for i in range(self.n_outputs)]
        # now 'shape' (n_batch, n_targets) -> need to transpose.
        all_batch_targets = [[bpred[i] for bpred in all_batch_targets] for i in range(self.n_targets)]

        # now cat each prediction and target across the batch index.
        all_predictions = [torch.cat(x, dim=0) if x[0].shape != () else x[0] for x in all_batch_predictions]
        all_targets = [torch.cat(x, dim=0) for x in all_batch_targets]

        all_losses = [x.item() for x in self.eval_loss(*all_predictions, *all_targets)]
        self.eval_step_outputs.clear()  # free memory

        loss_dict = {name: value for name, value in zip(self.eval_names, all_losses)}

        self.log_dict({prefix+k:v for k,v in loss_dict.items()}, sync_dist=True)

        loss_dict = {prefix[:-1]:loss_dict}  # strip underscore from prefix.

        when = "Sanity Check" if self.trainer.sanity_checking else self.current_epoch
        # register metrics and push to controller
        out_ = self.metric_tracker.register_metrics(loss_dict, when=when)
        better_metrics, better_model, stopping_metric = out_
        self.stopping_metric = stopping_metric
        self.better_model = better_model
        self.metric_tracker.evaluation_print_better(loss_dict, better_metrics,_print=self.print)

        return better_model, stopping_metric

    def on_validation_epoch_end(self):
        better_model, stopping_metric = self._eval_epoch_end(prefix="valid_")
        if self.trainer.sanity_checking:
            return

        continue_training = self.controller.push_epoch(self.current_epoch, better_model, stopping_metric, print_=self.print)

        return

    def on_test_epoch_end(self):
        self._eval_epoch_end(prefix="test_")
        return


class HippynnControllerCallback(pl.Callback):
    def __init__(self, trainer, controller, datamodule):
        self.controller = controller
        self.datamodule = datamodule

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.stopping_metric = None
        pl_module.better_model = None

    def on_validation_epoch_end(self, trainer, pl_module):

        better_model = pl_module.better_model
        continue_training = self.controller.push_epoch(pl_module.current_epoch, better_model, stopping_metric)




class HippynnDataModule(pl.LightningDataModule):
    def __init__(self, database:Database, batch_size):
        super().__init__()
        self.database = database
        self.batch_size = batch_size

    def train_dataloader(self):
        return self.database.make_generator("train", "train", self.batch_size)

    def val_dataloader(self):
        return self.database.make_generator("valid", "eval", self.batch_size)

    def test_dataloader(self):
        return self.database.make_generator("test", "eval", self.batch_size)