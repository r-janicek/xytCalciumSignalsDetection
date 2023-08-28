
import logging

import numpy as np

import unet

logger = logging.getLogger(__name__)

def load_training():
    
    # Change this to load your training data.
    training_x = np.float32(unet.load_volume("/cvlabdata1/home/marquez/datasets/hipp/training/training*.tif")) / 255.0
    training_y = np.uint8(unet.load_volume("/cvlabdata1/home/marquez/datasets/hipp/training_labels_mit/training_labels*.tif"))
    training_y[training_y == 255] = 1
    
    # convert to [-1,1], used for weight calculation
    training_y_bin = np.zeros_like( training_y, dtype=np.float32 )
    training_y_bin[training_y == 0] = -1
    training_y_bin[training_y == 1] = 1
    
    return training_x, training_y, training_y_bin

def load_testing():
    raise NotImplementedError

def main_3d():
    
    training_x, training_y, training_y_bin = load_training()
    
    # unet_config = unet.UnetConfig(steps=4, ndims=3, two_sublayers=False) # 66
    # ((158, 158, 158), (66, 66, 66)) -- YES
    
    # unet_config = unet.UnetConfig(steps=3, ndims=3, two_sublayers=True, num_classes=2) # 60
    # ((188, 188, 188), (100, 100, 100)) -- NO # Too much memory
    # ((172, 172, 172), (84, 84, 84)) -- NO # Too much memory
    # ((148, 148, 148), (60, 60, 60)) -- YES
    
    unet_config = unet.UNetConfig(steps=3, ndims=3, two_sublayers=False, num_classes=2, upsampling_mode='sparse') # 106
    # ((198, 198, 198), (154, 154, 154)) -- NO # Too much memory
    # ((150, 150, 150), (106, 106, 106)) -- YES
    
    rng = np.random.RandomState(42)
    logger.info("Creating U-Net...")
    unet_clsf = unet.UNetClassifier(unet_config, rng=rng)
    
    logger.info("Creating trainer...")
    trainer = unet.basic_setup(unet_clsf,
                                [training_x], [training_y],
                                hint_patch_shape=(25, 25, 25))
    
    # we will compute the weights for samplign every 500 iterations
    everyIters = 500
    lr = 1e-5  # learning rate

    howManyLoops = 50000 // everyIters

    import IPython
    IPython.embed()

    from unet.utils import pad_for_unet
    for i in range(howManyLoops):
        trainer.train(everyIters, 10, solver_kwargs={'learning_rate': lr})

        pred = prediction.predict_in_blocks( unet_clsf, training_x, (45,45,45) )
        pred = pred[:,:,:,1] - pred[:,:,:,0]
        
        # weights come from the gradient of the loss wrt each u-net output
        # and their summed magnitudes. For binary classification it has a very
        # simple expression:
        # (we ignore weight labels here)
        weights = 1.0 / (1 + np.exp(training_y_bin*pred))

        # save weights and current prediction for debugging?
        # import nrrd
        # nrrd.write( trainer.save_path + "/weights.nrrd", np.transpose(weights, [2,1,0] ) )
        # nrrd.write( trainer.save_path + "/pred.nrrd", np.transpose(pred, [2,1,0] ) )

        # pad and update sampler weights
        weights = pad_for_unet(weights, unet_config)
        trainer.sampler.update_sampling_weights( [weights] )
        
    import IPython
    IPython.embed()


def predict2d( cnn, stack ):
    pred = np.zeros( stack.shape, dtype=np.float32 )
    from unet import prediction
    for i in range(pred.shape[0]):
        p = prediction.predict_in_blocks( cnn, stack[i,:,:], (256,256) )
        pred[i,:,:] = p[:,:,1] - p[:,:,0]
        print(i,pred.shape[0])
    return pred

def predict2d_reg( cnn, stack ):
    pred = np.zeros( stack.shape, dtype=np.float32 )
    from unet import prediction
    for i in range(pred.shape[0]):
        p = prediction.predict_in_blocks( cnn, stack[i,:,:], (512,512) )
        pred[i,:,:] = p
    return pred

def main_2d():
    
    training_x, training_y, training_y_bin = load_training()
    
    # Create the network
    rng = np.random.RandomState(42)
    logger.info("Creating U-Net...")
    unet_config = unet.UNetConfig(
                                steps=4,
                                ndims=2,
                                num_classes=2,
                                num_input_channels=1,
                                two_sublayers=False,
                                upsampling_mode='sparse')
    unet_clsf = unet.UNetClassifier(unet_config, rng=rng)
    
    # Create the trainer
    trainer = unet.basic_setup(unet_clsf,
                                training_x, training_y,
                                hint_patch_shape=(388, 388),
                                augment_data='all',
                                save_path="./savedmodels",
                                save_every=None, # If None, save at the end of every epoch
                                rng=np.random.RandomState())
    
    # we will compute the weights for samplign every 500 iterations
    everyIters = 1000
    lr = 1e-5  # learning rate

    howManyLoops = 50000 // everyIters
    
    
    import IPython
    IPython.embed()
    
    
    from unet.utils import pad_for_unet
    for i in range(howManyLoops):
        trainer.train(everyIters, 10, solver_kwargs={'learning_rate': lr})

        pred = predict2d( unet_clsf, training_x )
        weights = 1.0 / (1 + np.exp(training_y_bin*pred))

        # save weights for debugging?
        # import nrrd
        # nrrd.write( trainer.save_path + "/weights.nrrd", np.transpose(weights, [2,1,0] ) )
        # nrrd.write( trainer.save_path + "/pred.nrrd", np.transpose(pred, [2,1,0] ) )

        # update u-net weights
        weights = [pad_for_unet(w, unet_config) for w in weights]
        trainer.sampler.update_sampling_weights( weights )
        
    import IPython
    IPython.embed()


def main_2d_regr():
    
    training_x, training_y, training_y_bin = load_training()
    
    # Create the network
    rng = np.random.RandomState(42)
    logger.info("Creating U-Net...")
    unet_config = unet.UNetConfig(steps=4, ndims=2, num_classes=1,
                                    two_sublayers=False,
                                    upsampling_mode='sparse')
    unet_regr = unet.UNetRegressor(unet_config, rng=rng)
    
    # Create the trainer
    trainer = unet.basic_setup(unet_regr,
                                training_x, training_y_bin,
                                hint_patch_shape=(388, 388),
                                augment_data='all',
                                save_path="./savedmodels",
                                save_every=None, # If None, save at the end of every epoch
                                rng=np.random.RandomState())
    
    
    # we will compute the weights for samplign every 500 iterations
    everyIters = 1000
    lr = 1e-4  # learning rate

    howManyLoops = 50000 // everyIters
    
    
    import IPython
    IPython.embed()
    
    
    from unet.utils import pad_for_unet
    for i in range(howManyLoops):
        trainer.train(everyIters, 10, solver_kwargs={'learning_rate': lr})

        pred = predict2d_reg( unet_regr, training_x )
        weights = np.abs(training_y_bin - pred)  # loss gradient magnitude

        # save weights for debugging?
        import nrrd
        nrrd.write( trainer.save_path + "/weights.nrrd", np.transpose(weights, [2,1,0] ) )
        nrrd.write( trainer.save_path + "/pred.nrrd", np.transpose(pred, [2,1,0] ) )

        # update u-net weights
        weights = [pad_for_unet(w, unet_config) for w in weights]
        trainer.sampler.update_sampling_weights( weights )
        
    import IPython
    IPython.embed()
    

if __name__ == "__main__":
    np.set_printoptions(precision=4)
    unet.config_logger("/dev/null")
    # main_2d()
    # main_3d()
    main_2d_regr()
