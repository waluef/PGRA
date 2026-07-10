"""Early stopping."""

import copy

import torch


class EarlyStopping(object):
    def __init__(self, save_path, patience=50, mode="max",
                 min_delta=0.0, verbose=False):
        assert mode in ("max", "min"), "mode must be 'max' or 'min'"
        self.save_path = save_path
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)

        self._sign = 1.0 if mode == "max" else -1.0
        self.best_score = None
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False
        self._best_state = None

    def step(self, score, model, epoch=None):
        score = float(score)

        if self.mode == "max" and score >= 1.0:
            self._commit(score, model, epoch)
            self.should_stop = True
            return True

        if self._is_improvement(score):
            self._commit(score, model, epoch)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print("[EarlyStopping] {}/{} epochs without improvement".format(
                    self.counter, self.patience))
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def _is_improvement(self, score):
        if self.best_score is None:
            return True
        return self._sign * (score - self.best_score) > self.min_delta

    def _commit(self, score, model, epoch):
        self.best_score = score
        self.best_epoch = -1 if epoch is None else int(epoch)
        self._best_state = copy.deepcopy(model.state_dict())
        torch.save(self._best_state, self.save_path)

    def load_best(self, model):
        if self._best_state is None:
            raise RuntimeError("EarlyStopping has no checkpoint to restore.")
        model.load_state_dict(self._best_state)
        return model

    def __call__(self, score, model):
        self.step(score, model)
        return self.counter

    @property
    def early_stop(self):
        return self.should_stop

    @property
    def best_acc(self):
        return self.best_score
