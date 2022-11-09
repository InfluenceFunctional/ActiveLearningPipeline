from argparse import Namespace
from models import modelNet
from querier import Querier
from sampler import Sampler
from oracle import Oracle
from utils import (
    namespace2dict,
    printRecord,
    numpy2python,
    bcolors,
    numbers2letters,
    filterOutputs,
    binaryDistance,
    get_n_params,
)
from torch.utils import data
import torch.nn.functional as F
import torch
import pandas as pd

import time
import sys
import yaml
import numpy as np


class ActiveLearning:
    def __init__(self, config, logger=None, data_to_log=None, episode=None):
        self.pipeIter = None
        self.episode = episode
        self.data_to_log = data_to_log
        self.comet = logger
        self.config = config
        self.workdir = config.workdir
        self.runNum = self.config.run_num
        self.pipeIter = 0
        self.oracle = Oracle(
            self.config
        )  # oracle needs to be initialized to initialize toy datasets
        self.querier = Querier(self.config)  # might as well initialize the querier here
        self.oracle.initializeDataset()
        self.getModelSize()

        # Save YAML config
        with open(self.config.workdir + "/config.yml", "w") as f:
            yaml.dump(numpy2python(namespace2dict(self.config)), f, default_flow_style=False)

        self.config.dataset_size = self.config.dataset.init_length

        if self.config.dataset.type == "toy":
            self.sampleOracle()  # use the oracle to pre-solve the problem for future benchmarking

        self.testMinima = []  # best test loss of models, for each iteration of the pipeline
        self.bestScores = []  # best optima found by the sampler, for each iteration of the pipeline

        t0 = time.time()
        self.retrainModels()
        if self.config.debug:
            printRecord(f"Initial training took {int(time.time() - t0)} seconds")

        t0 = time.time()
        self.getModelState()  # run energy-only sampling and create model state dict
        self.getDatasetState()
        if self.config.debug:
            printRecord(f"Model state calculation took {int(time.time() - t0)} seconds")

    def iterate(self, action, terminal):
        """
        run one iteration of the pipeline - train model, sample sequences, select sequences, consult oracle
        :return:
        """
        self.pipeIter += 1
        if terminal == 0:  # skip querying if this is our final pipeline iteration
            if self.config.debug:
                printRecord(
                    f"Starting pipeline iteration #{bcolors.FAIL}{int(self.pipeIter + 1)}{bcolors.ENDC}"
                )
            t0 = time.time()
            query = self.querier.buildQuery(
                self.model,
                self.stateDict,
                action=action,
                comet=self.comet,
            )  # pick Samples to be scored
            if self.config.debug:
                printRecord(f"Query generation took {int(time.time() - t0)} seconds")

            t0 = time.time()
            energies = self.oracle.score(query)  # score Samples
            if self.config.debug:
                printRecord(f"Oracle scoring took {int(time.time() - t0)} seconds")
                printRecord(
                    "Oracle scored"
                    + bcolors.OKBLUE
                    + " {} ".format(len(energies))
                    + bcolors.ENDC
                    + "queries with average score of"
                    + bcolors.OKGREEN
                    + " {:.3f}".format(np.average(energies))
                    + bcolors.ENDC
                    + " and minimum score of {:.3f}".format(np.amin(energies))
                )

            self.updateDataset(query, energies)  # add scored Samples to dataset

            if self.comet and (
                "Query_Energies" in self.data_to_log
            ):  # report query scores to comet
                self.comet.log_histogram_3d(energies, name="query energies", step=self.pipeIter)

            t0 = time.time()
            self.retrainModels()
            if self.config.debug:
                printRecord("Retraining took {} seconds".format(int(time.time() - t0)))

            t0 = time.time()
            self.getModelState(terminal)  # run energy-only sampling and create model state dict
            self.getDatasetState()
            if self.config.debug:
                printRecord("Model state calculation took {} seconds".format(int(time.time() - t0)))

    def getModelState(self, terminal=0):
        """
        sample the model
        report on the status of dataset
        report on best scores according to models
        report on model confidence
        :return:
        """

        # run the sampler
        self.loadEstimatorEnsemble()
        if terminal:  # use the query-generating sampler for terminal iteration
            sampleDict = self.querier.runSampling(
                self.model, scoreFunction=[1, 0], al_iter=self.pipeIter
            )  # sample existing optima using standard sampler
        else:  # use a cheap sampler for mid-run model state calculations
            sampleDict = self.querier.runSampling(
                self.model, scoreFunction=[1, 0], al_iter=self.pipeIter, method_overwrite="random"
            )  # sample existing optima cheaply with random + annealing

        sampleDict = filterOutputs(sampleDict, self.config.debug)

        # we used to do clustering here, now strictly argsort direct from the sampler
        sort_inds = np.argsort(sampleDict["energies"])  # sort by energy
        samples = sampleDict["samples"][sort_inds][
            : self.config.querier.model_state_size
        ]  # top-k samples from model state run
        energies = sampleDict["energies"][sort_inds][: self.config.querier.model_state_size]
        uncertainties = sampleDict["uncertainties"][sort_inds][
            : self.config.querier.model_state_size
        ]

        # get distances to relevant datasets
        internalDist, datasetDist, randomDist = self.getDataDists(samples)
        self.getModelStateReward(energies, uncertainties)

        self.stateDict = {
            "test loss": np.average(
                self.testMinima
            ),  # losses are evaluated on standardized data, so we do not need to re-standardize here
            "test std": np.sqrt(np.var(self.testMinima)),
            "all test losses": self.testMinima,
            "best energies": energies,  # these are already standardized #(energies - self.model.mean) / self.model.std, # standardize according to dataset statistics
            "best uncertanties": uncertainties,  # these are already standardized #uncertainties / self.model.std,
            "best samples": samples,
            "best samples internal diff": internalDist,
            "best samples dataset diff": datasetDist,
            "best samples random set diff": randomDist,
            "clustering cutoff": self.config.al.minima_dist_cutoff,  # could be a learned parameter
            "n proxy models": self.config.proxy.ensemble_size,
            "iter": self.pipeIter,
            "budget": self.config.al.n_iter,
            "model state reward": self.model_state_reward,
        }
        if self.config.debug:
            printRecord(
                "%d " % self.config.proxy.ensemble_size
                + f"Model ensemble training converged with average test loss of {bcolors.OKCYAN}%.5f{bcolors.ENDC}"
                % np.average(np.asarray(self.testMinima[-self.config.proxy.ensemble_size :]))
                + f" and std of {bcolors.OKCYAN}%.3f{bcolors.ENDC}"
                % (np.sqrt(np.var(self.testMinima[-self.config.proxy.ensemble_size :])))
            )
            printRecord(
                "Model state contains {} samples".format(self.config.querier.model_state_size)
                + " with minimum energy"
                + bcolors.OKGREEN
                + " {:.2f},".format(np.amin(energies))
                + bcolors.ENDC
                + " average energy"
                + bcolors.OKGREEN
                + " {:.2f},".format(np.average(energies[: self.config.querier.model_state_size]))
                + bcolors.ENDC
                + " and average std dev"
                + bcolors.OKCYAN
                + " {:.2f}".format(
                    np.average(uncertainties[: self.config.querier.model_state_size])
                )
                + bcolors.ENDC
            )
            printRecord(
                "Best sample in model state is {}".format(
                    numbers2letters(samples[np.argmin(energies)])
                )
            )
            printRecord(
                "Sample average mutual distance is "
                + bcolors.WARNING
                + "{:.2f} ".format(np.average(internalDist))
                + bcolors.ENDC
                + "dataset distance is "
                + bcolors.WARNING
                + "{:.2f} ".format(np.average(datasetDist))
                + bcolors.ENDC
                + "and overall distance estimated at "
                + bcolors.WARNING
                + "{:.2f}".format(np.average(randomDist))
                + bcolors.ENDC
            )

        if (
            self.config.al.large_model_evaluation
        ):  # we can quickly check the test error against a huge random dataset
            self.largeModelEvaluation()
            if self.comet and ("Proxy_Evaluation_Loss" in self.data_to_log):
                self.comet.log_metric(
                    name="proxy loss on best 10% of large random dataset",
                    value=self.bottomTenLoss[0],
                    step=self.pipeIter,
                )
                self.comet.log_metric(
                    name="proxy loss on large random dataset",
                    value=self.totalLoss[0],
                    step=self.pipeIter,
                )

        if self.pipeIter == 0:  # if it's the first round, initialize, else, append
            self.stateDictRecord = [self.stateDict]
        else:
            self.stateDictRecord.append(self.stateDict)

        if self.comet and ("Model_State_Calculation" in self.data_to_log):
            self.comet.log_histogram_3d(
                sampleDict["energies"],
                name="model state total sampling run energies",
                step=self.pipeIter,
            )
            self.comet.log_histogram_3d(
                sampleDict["uncertainties"],
                name="model state total sampling run std deviations",
                step=self.pipeIter,
            )
            self.comet.log_histogram_3d(
                energies[: self.config.querier.model_state_size],
                name="model state energies",
                step=self.pipeIter,
            )
            self.comet.log_histogram_3d(
                uncertainties[: self.config.querier.model_state_size],
                name="model state std deviations",
                step=self.pipeIter,
            )
            self.comet.log_histogram_3d(
                internalDist, name="model state internal distance", step=self.pipeIter
            )
            self.comet.log_histogram_3d(
                datasetDist, name="model state distance from dataset", step=self.pipeIter
            )
            self.comet.log_histogram_3d(
                randomDist, name="model state distance from large random sample", step=self.pipeIter
            )
            self.comet.log_histogram_3d(
                self.testMinima[-1], name="proxy model test minima", step=self.pipeIter
            )
            self.logTopK(sampleDict, prefix="Model state ")

    def getModelStateReward(self, bestEns, bestStdDevs):
        """
        print the performance of the learner against a known best answer
        :param bestEns:
        :param bestVars:
        :return:
        """
        # get the best results in the standardized basis
        best_ens_standardized = (bestEns - self.model.mean) / self.model.std
        standardized_standard_deviations = bestStdDevs / self.model.std
        adjusted_standardized_energies = (
            best_ens_standardized + standardized_standard_deviations
        )  # consider std dev as an uncertainty envelope and take the high end
        best_standardized_adjusted_energy = np.amin(adjusted_standardized_energies)

        # convert to raw outputs basis
        adjusted_energies = bestEns + bestStdDevs
        best_adjusted_energy = np.amin(adjusted_energies)  # best energy, adjusted for uncertainty
        if self.pipeIter == 0:
            self.model_state_reward = 0  # first iteration - can't define a reward
            self.model_state_cumulative_reward = 0
            self.model_state_reward_list = np.zeros(self.config.al.n_iter)
            self.model_state_prev_iter_best = [best_adjusted_energy]
        else:  # calculate reward using current standardization
            stdprev_iter_best = (
                self.model_state_prev_iter_best[-1] - self.model.mean
            ) / self.model.std
            self.model_state_reward = -(
                best_standardized_adjusted_energy - stdprev_iter_best
            )  # reward is the delta between variance-adjusted energies in the standardized basis (smaller is better)
            self.model_state_reward_list[self.pipeIter] = self.model_state_reward
            self.model_state_cumulative_reward = sum(self.model_state_reward_list)
            self.model_state_prev_iter_best.append(best_adjusted_energy)
            if self.config.debug:
                printRecord(
                    "Iteration best uncertainty-adjusted result = {:.3f}, previous best = {:.3f}, reward = {:.3f}, cumulative reward = {:.3f}".format(
                        best_adjusted_energy,
                        self.model_state_prev_iter_best[-2],
                        self.model_state_reward,
                        self.model_state_cumulative_reward,
                    )
                )

        if (
            self.config.dataset.type == "toy"
        ):  # if it's  a toy dataset, report the cumulative performance against the known minimum
            # stdTrueMinimum = (self.trueMinimum - self.model.mean) / self.model.std
            if self.pipeIter == 0:
                self.model_state_abs_score = [
                    1 - np.abs(self.trueMinimum - best_adjusted_energy) / np.abs(self.trueMinimum)
                ]
                self.model_state_cumulative_score = 0
            elif self.pipeIter > 0:
                # we will compute the distance from our best answer to the correct answer and integrate it over the number of samples in the dataset
                xaxis = (
                    self.config.dataset_size
                    + np.arange(0, self.pipeIter + 1) * self.config.al.queries_per_iter
                )  # how many samples in the dataset used for each
                self.model_state_abs_score.append(
                    1 - np.abs(self.trueMinimum - best_adjusted_energy) / np.abs(self.trueMinimum)
                )  # compute proximity to correct answer in standardized basis
                self.model_state_cumulative_score = np.trapz(
                    y=np.asarray(self.model_state_abs_score), x=xaxis
                )
                self.model_state_normed_cumulative_score = (
                    self.model_state_cumulative_score / xaxis[-1]
                )
                if self.config.debug:
                    printRecord(
                        "Total score is {:.3f} and {:.5f} per-sample after {} samples".format(
                            self.model_state_abs_score[-1],
                            self.model_state_normed_cumulative_score,
                            xaxis[-1],
                        )
                    )
            else:
                print("Error! Pipeline iteration cannot be negative")
                sys.exit()

            if self.comet and ("Model_State_Score" in self.data_to_log):
                self.comet.log_metric(
                    name=f"Ep {self.episode} - model state absolute score",
                    value=self.model_state_abs_score[-1],
                    step=self.pipeIter,
                )
                self.comet.log_metric(
                    name=f"Ep {self.episode} - model state cumulative score",
                    value=self.model_state_cumulative_score,
                    step=self.pipeIter,
                )
                self.comet.log_metric(
                    name=f"Ep {self.episode} - model state reward",
                    value=self.model_state_reward,
                    step=self.pipeIter,
                )

    def getDatasetState(self):
        """
        print the performance of the learner against a known best answer
        :param bestEns:
        :param bestVars:
        :return:
        """
        dataset = np.load(
            f"{self.config.workdir}/datasets/{self.config.dataset.oracle}.npy", allow_pickle=True
        ).item()
        energies = dataset["energies"]
        if self.config.debug:
            printRecord(
                "Best sample in dataset is {}".format(
                    numbers2letters(dataset["samples"][np.argmin(dataset["energies"])])
                )
            )

        best_energy = np.amin(energies)
        if self.pipeIter == 0:
            self.dataset_reward = 0  # first iteration - can't define a reward
            self.dataset_cumulative_reward = 0
            self.dataset_reward_list = np.zeros(self.config.al.n_iter)
            self.dataset_prev_iter_best = [best_energy]
        else:  # calculate reward using current standardization
            self.dataset_reward = (
                best_energy - self.dataset_prev_iter_best[-1]
            ) / self.dataset_prev_iter_best[
                -1
            ]  # reward is the delta between variance-adjusted energies in the standardized basis (smaller is better)
            self.dataset_reward_list[self.pipeIter] = self.dataset_reward
            self.dataset_cumulative_reward = sum(self.dataset_reward_list)
            self.dataset_prev_iter_best.append(best_energy)
            if self.config.debug:
                printRecord(
                    "Dataset evolution metrics = {:.3f}, previous best = {:.3f}, reward = {:.3f}, cumulative reward = {:.3f}".format(
                        best_energy,
                        self.dataset_prev_iter_best[-2],
                        self.dataset_reward,
                        self.dataset_cumulative_reward,
                    )
                )

        if (
            self.config.dataset.type == "toy"
        ):  # if it's  a toy dataset, report the cumulative performance against the known minimum
            # stdTrueMinimum = (self.trueMinimum - self.model.mean) / self.model.std
            if self.pipeIter == 0:
                self.dataset_abs_score = [
                    1 - np.abs(self.trueMinimum - best_energy) / np.abs(self.trueMinimum)
                ]
                self.dataset_cumulative_score = 0
            elif self.pipeIter > 0:
                # we will compute the distance from our best answer to the correct answer and integrate it over the number of samples in the dataset
                xaxis = (
                    self.config.dataset_size
                    + np.arange(0, self.pipeIter + 1) * self.config.al.queries_per_iter
                )  # how many samples in the dataset used for each
                self.dataset_abs_score.append(
                    1 - np.abs(self.trueMinimum - best_energy) / np.abs(self.trueMinimum)
                )  # compute proximity to correct answer in standardized basis
                self.dataset_cumulative_score = np.trapz(
                    y=np.asarray(self.dataset_abs_score), x=xaxis
                )
                self.dataset_normed_cumulative_score = self.dataset_cumulative_score / xaxis[-1]
                if self.config.debug:
                    printRecord(
                        "Dataset Total score is {:.3f} and {:.5f} per-sample after {} samples".format(
                            self.dataset_abs_score[-1],
                            self.dataset_normed_cumulative_score,
                            xaxis[-1],
                        )
                    )
            else:
                print("Error! Pipeline iteration cannot be negative")
                sys.exit()

            if self.comet and ("Dataset_State_Scores" in self.data_to_log):
                self.comet.log_metric(
                    name="dataset absolute score",
                    value=self.dataset_abs_score[-1],
                    step=self.pipeIter,
                )
                self.comet.log_metric(
                    name="dataset cumulative score",
                    value=self.dataset_cumulative_score,
                    step=self.pipeIter,
                )
                self.comet.log_metric(
                    name="dataset reward", value=self.dataset_reward, step=self.pipeIter
                )

    def retrainModels(self):
        testMins = []
        for i in range(self.config.proxy.ensemble_size):
            self.resetModel(i)  # reset between ensemble estimators EVERY ITERATION of the pipeline
            self.model.converge()  # converge model
            testMins.append(np.amin(self.model.err_te_hist))
            if self.comet and ("Proxy_Train_Loss" in self.data_to_log):
                tr_hist = self.model.err_tr_hist
                te_hist = self.model.err_te_hist
                epochs = len(te_hist)
                for i in range(epochs):
                    self.comet.log_metric(
                        "proxy train loss iter {}".format(self.pipeIter), step=i, value=tr_hist[i]
                    )
                    self.comet.log_metric(
                        "proxy test loss iter {}".format(self.pipeIter), step=i, value=te_hist[i]
                    )

        self.testMinima.append(testMins)

    def loadEstimatorEnsemble(self):
        """
        load all the trained models at their best checkpoints
        and initialize them in an ensemble model where they can all be queried at once
        :return:
        """
        ensemble = []
        for i in range(self.config.proxy.ensemble_size):
            self.resetModel(i)
            self.model.load(i)
            ensemble.append(self.model.model)

        del self.model
        self.model = modelNet(self.config, 0)
        self.model.loadEnsemble(ensemble)
        self.model.getMinF()

        # print('Loaded {} estimators'.format(int(self.config.proxy.ensemble_size)))

    def resetModel(self, ensembleIndex, returnModel=False):
        """
        load a new instance of the model with reset parameters
        :return:
        """
        try:  # if we have a model already, delete it
            del self.model
        except:
            pass
        self.model = modelNet(self.config, ensembleIndex)
        # printRecord(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getModelName(ensembleIndex))
        if returnModel:
            return self.model

    def getModelSize(self):
        self.model = modelNet(self.config, 0)
        nParams = get_n_params(self.model.model)
        if self.config.debug:
            printRecord("Proxy model has {} parameters".format(int(nParams)))
        del self.model

    def largeModelEvaluation(self):
        """
        if we are using a toy oracle, we should be able to easily get the test loss on a huge sample of the dataset
        :return:
        """
        self.loadEstimatorEnsemble()

        numSamples = min(
            int(1e3), self.config.dataset.dict_size**self.config.dataset.max_length // 100
        )  # either 1e5, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(
            save=False, returnData=True, customSize=numSamples
        )  # get large random dataset
        randomSamples = randomData["samples"]
        randomScores = randomData["energies"]

        sortInds = np.argsort(randomScores)  # sort randoms
        randomSamples = randomSamples[sortInds]
        randomScores = randomScores[sortInds]

        modelScores, modelStd = [[], []]
        sampleLoader = data.DataLoader(
            randomSamples,
            batch_size=self.config.proxy.mbsize,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        for i, testData in enumerate(sampleLoader):
            score, std_dev = self.model.evaluate(testData.float(), output="Both")
            modelScores.extend(score)
            modelStd.extend(std_dev)

        bestTenInd = numSamples // 10
        totalLoss = F.mse_loss(
            (torch.Tensor(modelScores).float() - self.model.mean) / self.model.std,
            (torch.Tensor(randomScores).float() - self.model.mean) / self.model.std,
        )  # full dataset loss (standardized basis)
        bottomTenLoss = F.mse_loss(
            (torch.Tensor(modelScores[:bestTenInd]).float() - self.model.mean) / self.model.std,
            (torch.Tensor(randomScores[:bestTenInd]).float() - self.model.mean) / self.model.std,
        )  # bottom 10% loss (standardized basis)

        if self.pipeIter == 0:  # if it's the first round, initialize, else, append
            self.totalLoss = [totalLoss]
            self.bottomTenLoss = [bottomTenLoss]
        else:
            self.totalLoss.append(totalLoss)
            self.bottomTenLoss.append(bottomTenLoss)

        printRecord(
            "Model has overall loss of"
            + bcolors.OKCYAN
            + " {:.5f}, ".format(totalLoss)
            + bcolors.ENDC
            + "best 10% loss of"
            + bcolors.OKCYAN
            + " {:.5f} ".format(bottomTenLoss)
            + bcolors.ENDC
            + "on {} toy dataset samples".format(numSamples)
        )

    def sampleOracle(self):
        """
        for toy models
        do global optimization directly on the oracle to find the true minimum
        :return:
        """
        if self.config.debug:
            printRecord("Asking toy oracle for the true minimum")

        self.model = "abc"
        gammas = np.logspace(
            self.config.mcmc.stun_min_gamma,
            self.config.mcmc.stun_max_gamma,
            self.config.mcmc.num_samplers,
        )
        mcmcSampler = Sampler(self.config, 0, [1, 0], gammas)
        if (self.config.dataset.oracle == "linear") or ("nupack" in self.config.dataset.oracle):
            sampleDict = mcmcSampler.sample(
                self.model, useOracle=True, nIters=100
            )  # do a tiny number of iters - the minimum is known
        else:
            sampleDict = mcmcSampler.sample(self.model, useOracle=True)  # do a genuine search

        bestMin = self.getTrueMinimum(sampleDict)
        if self.config.debug:
            printRecord(
                f"Sampling Complete! Lowest Energy Found = {bcolors.FAIL}%.3f{bcolors.ENDC}"
                % bestMin
                + " from %d" % self.config.mcmc.num_samplers
                + " sampling runs."
            )
            printRecord(
                "Best sample found is {}".format(
                    numbers2letters(sampleDict["samples"][np.argmin(sampleDict["energies"])])
                )
            )

        self.oracleRecord = sampleDict
        self.trueMinimum = bestMin

        if self.comet and ("Sample_Energies" in self.data_to_log):
            self.comet.log_histogram_3d(sampleDict["energies"], name="energies_true", step=0)

    def getTrueMinimum(self, sampleDict):

        if (
            self.config.dataset.oracle == "wmodel"
        ):  # w model minimum is always zero - even if we don't find it
            bestMin = 0
        else:
            bestMin = np.amin(sampleDict["energies"])

        if (
            "nupack" in self.config.dataset.oracle
        ):  # compute minimum energy for this length - for reweighting purposes
            goodSamples = (
                np.ones((4, self.config.dataset.max_length)) * 4
            )  # GCGC CGCG GGGCCC CCCGGG
            goodSamples[0, 0:-1:2] = 3
            goodSamples[1, 1:-1:2] = 3
            goodSamples[2, : self.config.dataset.max_length // 2] = 3
            goodSamples[3, self.config.dataset.max_length // 2 :] = 3
            min_nupack_ens = self.oracle.score(goodSamples)

        # append suggestions for known likely solutions
        if self.config.dataset.oracle == "linear":
            goodSamples = np.zeros(
                (4, self.config.dataset.max_length)
            )  # all of one class usually best
            goodSamples[0] = goodSamples[1] + 1
            goodSamples[1] = goodSamples[1] + 2
            goodSamples[2] = goodSamples[2] + 3
            goodSamples[3] = goodSamples[3] + 4
            ens = self.oracle.score(goodSamples)
            if np.amin(ens) < bestMin:
                bestMin = np.amin(ens)
                if self.config.debug:
                    printRecord("Pre-loaded minimum was better than one found by sampler")

        elif self.config.dataset.oracle == "nupack energy":
            if np.amin(min_nupack_ens) < bestMin:
                bestMin = np.amin(min_nupack_ens)
                if self.config.debug:
                    printRecord("Pre-loaded minimum was better than one found by sampler")

        elif self.config.dataset.oracle == "nupack pairs":
            goodSamples = (
                np.ones((4, self.config.dataset.max_length)) * 4
            )  # GCGC CGCG GGGCCC CCCGGG
            goodSamples[0, 0:-1:2] = 3
            goodSamples[1, 1:-1:2] = 3
            goodSamples[2, : self.config.dataset.max_length // 2] = 3
            goodSamples[3, self.config.dataset.max_length // 2 :] = 3
            ens = self.oracle.score(goodSamples)
            if np.amin(ens) < bestMin:
                bestMin = np.amin(ens)
                if self.config.debug:
                    printRecord("Pre-loaded minimum was better than one found by sampler")

        elif self.config.dataset.oracle == "nupack pins":
            max_pins = (
                self.config.dataset.max_length // 12
            )  # a conservative estimate - 12 bases per stable hairpin
            if max_pins < bestMin:
                bestMin = max_pins
                if self.config.debug:
                    printRecord("Pre-run guess was better than one found by sampler")

        elif self.config.dataset.oracle == "nupack open loop":
            biggest_loop = (
                self.config.dataset.max_length - 8
            )  # a conservative estimate - 8 bases for the stem (10 would be more conservative) and the rest are open
            if biggest_loop < bestMin:
                bestMin = biggest_loop
                if self.config.debug:
                    printRecord("Pre-run guess was better than one found by sampler")

        elif self.config.dataset.oracle == "nupack motif":
            bestMin = -1  # 100% agreement is the best possible

        return bestMin

    def saveOutputs(self):
        """
        save config and outputs in a dict
        :return:
        """
        outputDict = {}
        outputDict["config"] = Namespace(**dict(vars(self.config)))
        if "comet" in outputDict["config"]:
            del outputDict["config"].comet
        outputDict["state dict record"] = self.stateDictRecord
        outputDict["model state rewards"] = self.model_state_reward_list
        outputDict["dataset rewards"] = self.dataset_reward_list
        if self.config.al.large_model_evaluation:
            outputDict["big dataset loss"] = self.totalLoss
            outputDict["bottom 10% loss"] = self.bottomTenLoss
        if self.config.dataset.type == "toy":
            outputDict["oracle outputs"] = self.oracleRecord
            if self.pipeIter > 1:
                outputDict["model state score record"] = self.model_state_abs_score
                outputDict["model state cumulative score"] = (self.model_state_cumulative_score,)
                outputDict[
                    "model state per sample cumulative score"
                ] = self.model_state_normed_cumulative_score
                outputDict["dataset score record"] = self.dataset_abs_score
                outputDict["dataset cumulative score"] = (self.dataset_cumulative_score,)
                outputDict[
                    "dataset per sample cumulative score"
                ] = self.dataset_normed_cumulative_score
        np.save("outputsDict", outputDict)  # Need to use .item() when loading to get the data

    def updateDataset(self, oracleSequences, oracleScores):
        """
        loads dataset, appends new datapoints from oracle, and saves dataset
        :param params: model parameters
        :param oracleSequences: sequences which were sent to oracle
        :param oracleScores: scores of sequences sent to oracle
        :return: n/a
        """
        dataset = np.load(
            f"{self.config.workdir}/datasets/{self.config.dataset.oracle}.npy", allow_pickle=True
        ).item()
        dataset["samples"] = np.concatenate((dataset["samples"], oracleSequences))
        dataset["energies"] = np.concatenate((dataset["energies"], oracleScores))

        self.config.dataset_size = len(dataset["samples"])
        if self.config.debug:
            printRecord(
                f"Added{bcolors.OKBLUE}{bcolors.BOLD} %d{bcolors.ENDC}" % int(len(oracleSequences))
                + " to the dataset, total dataset size is"
                + bcolors.OKBLUE
                + " {}".format(int(len(dataset["samples"])))
                + bcolors.ENDC
            )
            printRecord(
                bcolors.UNDERLINE
                + "====================================================================="
                + bcolors.ENDC
            )
        np.save(f"{self.config.workdir}/datasets/{self.config.dataset.oracle}", dataset)
        np.save(
            f"{self.config.workdir}/datasets/{self.config.dataset.oracle}_iter_{self.pipeIter}",
            dataset,
        )

        if self.comet and ("Dataset_Energies" in self.data_to_log):

            self.logTopK(
                dataset, prefix="Dataset"
            )  # log statistics on top K samples from the dataset
            self.comet.log_histogram_3d(
                dataset["energies"], name="dataset energies", step=self.pipeIter
            )
            dataset2 = dataset.copy()
            dataset2["samples"] = numbers2letters(dataset["samples"])
            self.comet.log_table(
                filename="dataset_at_iter_{}.csv".format(self.pipeIter),
                tabular_data=pd.DataFrame.from_dict(dataset2),
            )

    def logTopK(self, dataset, prefix, returnScores=False):
        if self.comet:
            self.comet.log_histogram_3d(
                dataset["energies"], name=prefix + " energies", step=self.pipeIter
            )
            idx_sorted = np.argsort(dataset["energies"])
            top_scores = []
            for k in [1, 10, 100]:
                topk_scores = dataset["energies"][idx_sorted[:k]]
                topk_samples = dataset["samples"][idx_sorted[:k]]
                top_scores.append(np.average(topk_scores))
                dist = binaryDistance(topk_samples, pairwise=False, extractInds=len(topk_samples))
                self.comet.log_metric(
                    prefix + f" mean top-{k} energies", np.mean(topk_scores), step=self.pipeIter
                )
                self.comet.log_metric(
                    prefix + f" std top-{k} energies", np.std(topk_scores), step=self.pipeIter
                )
                self.comet.log_metric(
                    prefix + f" mean dist top-{k}", np.mean(dist), step=self.pipeIter
                )

            if returnScores:
                return np.asarray(top_scores)

    def getDataDists(self, samples):
        """
        compute average binary distances between a set of samples and
        1 - itself
        2 - the training dataset
        3 - a large random sample
        :param samples:
        :return:
        """
        # training dataset
        dataset = np.load(
            f"{self.config.workdir}/datasets/{self.config.dataset.oracle}.npy", allow_pickle=True
        ).item()
        dataset = dataset["samples"]

        # large, random sample
        numSamples = min(
            int(1e3), self.config.dataset.dict_size**self.config.dataset.max_length // 100
        )  # either 1eX, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(
            save=False, returnData=True, customSize=numSamples
        )  # get large random dataset
        randomSamples = randomData["samples"]

        internalDist = binaryDistance(
            samples, self.config.dataset.dict_size, pairwise=False, extractInds=len(samples)
        )
        datasetDist = binaryDistance(
            np.concatenate((samples, dataset)),
            self.config.dataset.dict_size,
            pairwise=False,
            extractInds=len(samples),
        )
        randomDist = binaryDistance(
            np.concatenate((samples, randomSamples)),
            self.config.dataset.dict_size,
            pairwise=False,
            extractInds=len(samples),
        )

        return internalDist, datasetDist, randomDist
