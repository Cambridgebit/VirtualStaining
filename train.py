"""Training entry point with streamlined defaults.

Key improvements:
- Uses new BaseOptions defaults: GPU auto-detection, optional config file,
  automatic dataset discovery, and sensible experiment naming.
- Disables Visdom by default (enable with --display_id > 0).
"""
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options (with auto defaults)
    dataset = create_dataset(opt)  # primary dataset
    dataset2 = create_dataset(opt) # secondary stream (paired iteration)
    dataset_size = len(dataset)    # number of images
    opt.data_size = dataset_size

    model = create_model(opt)      # create and initialize model
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # displays/saves images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations
    optimize_time = 0.1

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # ensure HTML updates at least once per epoch

        dataset.set_epoch(epoch)
        try:
            dataset2.set_epoch(epoch)
        except AttributeError:
            pass

        for i, (data, data2) in enumerate(zip(dataset, dataset2)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            opt.counter = total_iters
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data, data2)
                # model.data_dependent_initialize()
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data, data2)  # unpack data from dataset(s) and apply preprocessing
            # model.set_input(data)
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
