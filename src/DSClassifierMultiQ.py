# coding=utf-8
import logging
import time
import numpy as np
import pandas as pd
import torch
from sklearn.base import ClassifierMixin

logger = logging.getLogger(__name__)

from .DSModelMultiQ import DSModelMultiQ

from .config import max_iter

class DSClassifierMultiQ(ClassifierMixin):
    """
    Implementation of Classifier based on DSModel
    """
    def __init__(self, num_classes, lr=0.005, max_iter=max_iter, min_iter=2, min_dloss=0.0001, optim="adam", lossfn="MSE",
                 debug_mode=False, step_debug_mode=False, batch_size=4000, num_workers=1,
                 precompute_rules=False, device="cpu", force_precompute=False,
                 maf_method="random", data=None, add_in_between_rules=False):
        """
        Creates the classifier and the DSModel (accesible in attribute model)
        :param lr: Learning rate
        :param max_iter: Maximun number of epochs in training
        :param min_dloss: Minium variation of loss to consider converged
        :param optim: [ adam | sgd ] Optimization Method
        :param lossfn: [ CE | MSE ] Loss function
        :param debug_mode: Enables debug in training (prtinting and output metrics)
        :param device: [ cpu | cuda | mps ] Device to use by pytorch
        :param force_precompute: Forces precomputation of rules, could use too much RAM
        :param maf_method: Method to generate random mass assignment functions (random | kmeans), default random
        """
        self.k = num_classes
        self.lr = lr
        self.optim = optim
        self.lossfn = lossfn
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        if self.device.type == "cuda" or self.device.type == "mps":
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                logger.warning("GPU acceleration is not available, using CPU")
                self.device = torch.device("cpu")
            else: # With GPU more workers may throw error
                self.num_workers = 0
        self.min_dJ = min_dloss
        self.balance_class_data = False
        self.debug_mode = debug_mode
        self.step_debug_mode = step_debug_mode
        self.maf_method = maf_method
        self.data = data # need for kmeans method
        self.add_in_between_rules = add_in_between_rules
        self.model = DSModelMultiQ(num_classes, precompute_rules=precompute_rules,
                                    device=self.device, force_precompute=force_precompute,
                                    maf_method=self.maf_method, data=self.data,
                                    add_in_between_rules=self.add_in_between_rules).to(self.device)
        self.classes_ = [k for k in range(self.k)]

    def fit(self, X, y, add_single_rules=False, single_rules_breaks=2, add_mult_rules=False, column_names=None, **kwargs):
        """
        Fits the model masses using gradient descent optimization
        :param X: Features for training
        :param y: Labels of features
        :param add_single_rules: Generates single rules
        :param single_rules_breaks: Single rule breaks number
        :param add_mult_rules: Generates multiplication pair rules
        :param kwargs: In case of debugging, parameters of optimize
        """
        if add_single_rules:
            self.model.generate_statistic_single_rules(X, breaks=single_rules_breaks, column_names=column_names)
        if add_mult_rules:
            self.model.generate_mult_pair_rules(X, column_names=column_names)

        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise RuntimeError("Unknown optimizer %s" % self.optim)

        if self.lossfn == "CE":
            criterion = torch.nn.CrossEntropyLoss()
        elif self.lossfn == "MSE":
            criterion = torch.nn.MSELoss()
        else:
            raise RuntimeError("Unknown loss function %s" % self.lossfn)

        # Add index to X
        X = np.insert(X, 0, values=np.arange(0, len(X)), axis=1)

        return self._optimize(X, y, optimizer, criterion,
                              debug=self.debug_mode, step_debug=self.step_debug_mode, **kwargs)

    def _optimize(self, X, y, optimizer, criterion, *,
                  debug=False, step_debug=False,
                  print_init_model=False, print_final_model=False, print_time=True,
                  print_partial_time=False, print_every_epochs=None, print_least_loss=True,
                  return_partial_dt=False, disable_all_print=False, print_epoch_progress=False):

        if step_debug:
            debug = True
            print_every_epochs = 1
            print_final_model = True
            print_partial_time = True
            print_time = True
            print_least_loss = True
            print_init_model = True

        if debug and disable_all_print:
            print_every_epochs = None
            print_final_model = False
            print_partial_time = False
            print_time = False
            print_least_loss = False
            print_init_model = False

        if debug:
            logger.info("Optimization started")
            if print_init_model:
                logger.debug(self.model)

        losses = []
        masses = [] if step_debug else None
        dt_forward = dt_loss = dt_optim = dt_norm = 0
        ti = time.time()

        self.model.train()
        self.model.clear_rmap()

        Xt = torch.Tensor(X).to(self.device)
        if self.lossfn == "CE":
            yt = torch.LongTensor(y).to(self.device)
        else:
            yt = torch.nn.functional.one_hot(torch.LongTensor(y).to(self.device), self.k).float()

        dataset = torch.utils.data.TensorDataset(Xt, yt)
        N = len(dataset)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                                   num_workers=self.num_workers, pin_memory=False)

        epoch = 0
        for epoch in range(self.max_iter):
            if debug and print_every_epochs is not None and epoch % print_every_epochs == 0:
                logger.debug("Processing epoch\t%d\t%.4f" % (epoch + 1, losses[-1] if len(losses) > 0 else 1))

            acc_loss = 0
            if debug and print_epoch_progress:
                acc_n = 0

            for Xi, yi in train_loader:
                if step_debug:
                    masses.append([mi.detach().tolist() for mi in self.model.parameters()])

                ni = len(yi)
                if debug and print_epoch_progress:
                    acc_n += ni
                    logger.debug("%d%% batch progress" % (100*acc_n/N))

                if debug:
                    tq = time.time()

                optimizer.zero_grad()
                y_pred = self.model.forward(Xi)

                if debug:
                    dt_forward += time.time() - tq
                    tq = time.time()

                loss = criterion(y_pred, yi)

                if np.isnan(loss.data.item()) or not np.isfinite(loss.data.item()):
                    if debug:
                        logger.error("Model: %s", self.model)
                        logger.error("y_pred: %s", y_pred)
                        logger.error("y_true: %s", yi)
                        logger.error("loss: %s", loss)
                    raise RuntimeError("Loss is NaN or Infinity")

                loss.backward()

                if debug:
                    dt_loss += time.time() - tq
                    tq = time.time()

                optimizer.step()

                if debug:
                    dt_optim += time.time() - tq
                    tq = time.time()

                self.model.normalize()

                if debug:
                    dt_norm += time.time() - tq

                acc_loss += loss.data.item() * ni / N

            losses.append(acc_loss)
            if epoch > self.min_iter and losses[-2] - acc_loss < self.min_dJ:
                break

        if not debug:
            return losses, epoch

        dt = time.time() - ti
        if print_time:
            logger.info("Training time: %.2fs, epochs: %d" % (dt, epoch + 1))

        if print_partial_time:
            logger.debug("├- Forward eval time:  %.3fs" % dt_forward)
            logger.debug("├- Loss backward time: %.3fs" % dt_loss)
            logger.debug("├- Optimization time:  %.3fs" % dt_optim)
            logger.debug("└- Normalization time: %.3fs" % dt_norm)

        if print_least_loss:
            logger.info("Least training loss reached: %.3f" % losses[-1])

        if print_final_model:
            logger.debug(self.model)

        if step_debug:
            masses = np.array(masses)
            return losses, epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm, masses
        elif return_partial_dt:
            return losses, epoch, dt, dt_forward, dt_loss, dt_optim, dt_norm
        else:
            return losses, epoch, dt

    def predict(self, X, one_hot=False):
        """
        Predict the classes for the feature vectors
        :param X: Feature vectors
        :param one_hot: If true, it is returned the score of belogning to each class
        :return: Classes for each feature vector
        """
        self.model.eval()
        self.model.clear_rmap()
        if "values" in dir(X):
            X = X.values
        X = np.insert(X, 0, values=np.arange(0, len(X)), axis=1)

        with torch.no_grad():
            Xt = torch.Tensor(X).to(self.device)
            if one_hot:
                return self.model(Xt).numpy()
            else:
                _, yt_pred = torch.max(self.model(Xt), 1)
                yt_pred = yt_pred.cpu().numpy()
                return yt_pred

    def predict_proba(self, X):
        """
        Predict the score of belogning to all classes
        :param X: Feature vector
        :return: Class scores for each feature vector
        """
        return self.predict(X, one_hot=True)

    def predict_explain(self, x):
        """
        Predict the score of belogning to each class and give an explanation of that decision
        :param x: A single Feature vectors
        :return:
        """
        pred = self.predict_proba([x])[0]
        cls = np.argmax(pred)
        rls, prds = self.model.get_rules_by_instance(x, order_by=cls)

        # String interpretation
        builder = "DS Model predicts class %d\n" % (cls + 1)
        cols = ["rule"]
        for i in range(len(pred)):
            builder += " Class %d: \t%.3f\n" % (i+1, pred[i])
            cols.append("mass_class_" + str(i+1))
        cols.append("uncertainty")

        df_rls = pd.DataFrame(rls)
        prds = [str(p) for p in prds]
        df_rls.insert(0, "rule", prds)
        df_rls = df_rls.loc[::-1].reset_index(drop=True)
        df_rls.columns = cols

        return pred, cls, df_rls, builder

    def print_most_important_rules(self, classes=None, threshold=0.2):
        return self.model.print_most_important_rules(classes, threshold)

    def find_most_important_rules(self, classes=None, threshold=0.2):
        return self.model.find_most_important_rules(classes, threshold)
