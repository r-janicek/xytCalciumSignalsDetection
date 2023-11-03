"""
Author: Pablo Marquéz-Neila
Modified by: Prisca Dotti
Last modified: 12/10/2023
"""

import logging
import os
import os.path
import time

import numpy as np
import torch

from . import summary

__all__ = ["TrainingManager", "managed_objects"]

logger = logging.getLogger(__name__)


def _managed_object(name, obj, filename_template="{name}_{{iter:06d}}.{extension}"):
    if hasattr(obj, "load_state_dict") and hasattr(obj, "state_dict"):
        # Torch module
        import torch

        return (
            obj,
            filename_template.format(name=name, extension="pth"),
            lambda obj, filename: torch.save(obj.state_dict(), filename),
            lambda obj, filename: obj.load_state_dict(torch.load(filename)),
        )
    elif hasattr(obj, "save_state") and hasattr(obj, "load_state"):
        return (
            obj,
            filename_template.format(name=name, extension="npz"),
            lambda obj, filename: obj.save_state(filename),
            lambda obj, filename: obj.load_state(filename),
        )

    raise ValueError("Unknown object type {}".format(obj))


def managed_objects(objects, filename_template="{name}_{{iter:06d}}.{extension}"):
    return [_managed_object(k, v, filename_template) for k, v in objects.items()]


def _write_results(summary_writer, prefix, results, niter):
    # TODO: Add dummy types to allow users to specify types

    for k, v in results.items():
        tag = "{}/{}".format(prefix, k)
        if isinstance(v, dict):
            summary_writer.add_scalars(tag, v, niter)
        elif isinstance(v, str):
            summary_writer.add_text(tag, v, niter)
        elif isinstance(v, np.ndarray) and v.ndim == 3:
            summary_writer.add_image(tag, v, niter)
        elif isinstance(v, np.ndarray) and v.ndim == 2:
            summary_writer.add_image(tag, v, niter, dataformats="HW")
        elif isinstance(v, torch.Tensor):  # new
            summary_writer.add_scalar(tag, v.item(), niter)
        elif not hasattr(v, "__iter__"):
            summary_writer.add_scalar(tag, v, niter)
        else:
            summary_writer.register(tag, v, niter)


class TrainingManager:
    def __init__(
        self,
        training_step,
        save_every=None,
        save_path=None,
        load_path=None,
        # List of tuples (object, filename_template, save_function, load_function)
        managed_objects=None,
        test_function=None,
        test_every=None,
        plot_function=None,
        plot_every=100,
        summary_writer=None,
    ):
        self.training_step = training_step
        self.iter = 0

        self.save_path = save_path
        self.load_path = load_path

        self.save_every = save_every
        self.managed_objects = managed_objects or []
        self.saved_at = set()

        if summary_writer is None:
            # Summary for registering data
            summary_writer = summary.Summary()
            # Add the summary to the list of managed objects.
            self.managed_objects.append(
                (
                    summary_writer,
                    "summary_{iter:06d}.npz",
                    summary.save_npz,
                    summary.load_npz,
                )
            )
        self.summary = summary_writer

        self.test_function = test_function
        self.test_every = test_every

        self.plot_function = plot_function
        self.plot_every = plot_every

    def save(self):
        if self.save_path is None:
            return

        if self.iter in self.saved_at:
            # Do not save same model twice
            return

        os.makedirs(self.save_path, exist_ok=True)

        for obj, filename_template, save_function, _ in self.managed_objects:
            filename = filename_template.format(iter=self.iter)
            filename = os.path.join(self.save_path, filename)

            logger.info("Saving '{}'...".format(filename))
            save_function(obj, filename)

        self.saved_at.add(self.iter)

    def load(self, niter):
        if self.load_path is None:
            self.load_path = self.save_path

        if self.load_path is None:
            raise ValueError(
                "`save_path` and `load_path` not set; cannot load a previous state"
            )

        for obj, filename_template, _, load_function in self.managed_objects:
            filename = filename_template.format(iter=niter)
            filename = os.path.join(self.load_path, filename)

            logger.info("Loading '{}'...".format(filename))
            load_function(obj, filename)

        self.iter = niter

    def run_validation(self):
        if self.test_function is None:
            return

        logger.info("Validating network at iteration {}...".format(self.iter))

        test_output = self.test_function(self.iter)

        if "loss" in test_output:
            logger.info("\tValidation loss: {:.4g}".format(test_output["loss"]))

        _write_results(self.summary, "testing", test_output, self.iter)

    def train(self, num_iters, print_every=0, maxtime=np.inf):
        tic = time.process_time()
        time_elapsed = 0

        for _ in range(num_iters):
            step_output = self.training_step(self.iter)

            time_elapsed = time.process_time() - tic

            # Register data
            _write_results(self.summary, "training", step_output, self.iter)

            # Check for nan
            loss = step_output["loss"]
            if np.any(np.isnan(loss)):
                logger.error("Last loss is nan! Training diverged!")
                break

            # logger.info(info)
            if print_every and self.iter % print_every == 0:
                logger.info("Iteration {}...".format(self.iter))
                logger.info("\tTraining loss: {:.4g}".format(loss))
                logger.info("\tTime elapsed: {:.2f}s".format(time_elapsed))

            self.iter += 1

            # Validation
            if self.test_every and self.iter % self.test_every == 0:
                self.run_validation()

            # Plot
            if (
                self.plot_function
                and self.plot_every
                and self.iter % self.plot_every == 0
            ):
                self.plot_function(self.iter, self.summary)

            # Save model and solver
            if self.save_every and self.iter % self.save_every == 0:
                self.save()

            if time_elapsed > maxtime:
                logger.info("Maximum time reached!")
                break
