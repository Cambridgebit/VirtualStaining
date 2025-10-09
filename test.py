"""Simplified test script for image-to-image translation.

Loads a trained model and runs inference on a dataset, saving results to HTML.
Defaults are set for deterministic single-thread, single-batch testing.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import _html_


if __name__ == '__main__':
    opt = TestOptions().parse()
    # Hard-code minimal, stable test settings
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1  # no Visdom; save to HTML only

    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    model = create_model(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    print('creating web directory', web_dir)
    webpage = _html_.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, (data, data2) in enumerate(zip(dataset, dataset2)):
        if i == 0:
            model.data_dependent_initialize(data, data2)
            model.setup(opt)
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:
            break

        model.set_input(data, data2)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    webpage.save()

