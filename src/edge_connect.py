import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy


class EdgeConnect():
    def __init__(self, config):
        self.config = config

        self.debug = False
        self.model_name = 'edge'
        self.edge_model = EdgeModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

    def load(self):
        self.edge_model.load()

    def save(self):
        self.edge_model.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.edge_model.train()

                images, images_gray, images_gray_stained, edges, edges_stained, masks = self.cuda(*items)

                # edge model
                # train
                outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, images_gray_stained, edges, edges_stained)

                # metrics
                precision, recall = self.edgeacc(edges, outputs)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))

                # backward
                self.edge_model.backward(gen_loss, dis_loss)
                iteration = self.edge_model.iteration


                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)

        self.edge_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, images_gray, images_gray_stained, edges, edges_stained, masks = self.cuda(*items)

            # edge model
            # eval
            outputs, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, images_gray_stained, edges, edges_stained)

            # metrics
            precision, recall = self.edgeacc(edges, outputs)
            logs.append(('precision', precision.item()))
            logs.append(('recall', recall.item()))

            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):
        self.edge_model.eval()

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, images_gray_stained, edges, edges_stained, masks = self.cuda(*items)
            index += 1

            # edge model
            outputs = self.edge_model(images_gray_stained, edges_stained)
            if self.debug:
                images = stitch_images(
                    self.postprocess(images_gray),
                    self.postprocess(images_gray_stained),
                    self.postprocess(edges_stained),
                    self.postprocess(edges),
                    self.postprocess(outputs),
                    img_per_row = 1
                )
                images.save(os.path.join(self.results_path, name))
            else:
                output = self.postprocess(outputs)[0]
                path = os.path.join(self.results_path, name)
                imsave(output, path)
            print(index, name)

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.edge_model.eval()

        items = next(self.sample_iterator)
        images, images_gray, images_gray_stained, edges, edges_stained, masks = self.cuda(*items)

        # edge model
        iteration = self.edge_model.iteration
        outputs = self.edge_model(images_gray_stained, edges_stained)

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(images_gray_stained),
            self.postprocess(edges),
            self.postprocess(outputs),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
