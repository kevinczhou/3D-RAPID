import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm
import scipy.signal
from mesoSfM import mesoSfM
import xarray as xr
import cv2
import os


class mcam3d(mesoSfM):
    def __init__(self, visitation_log_scale=None, do_not_copy=False, *args, **kwargs):
        # visitation_log_scale is the scale at which the visitation log was generated;
        # do_not_copy - if True, then don't create stack_tf, which is needed for creating tf.dataset; if this is not
        # needed, then set this to False (noting that if you try to create a tf.dataset, you'll get an error);
        super().__init__(*args, **kwargs)
        self.visitation_log_scale = visitation_log_scale
        if not do_not_copy:
            self.stack_tf = tf.constant(self.stack)  # sometimes autograph complains that this isn't a tf tensor (see
        # _generate_patched_dataset);
        self.TV_relaxation_coeff = None  # when computing MSE between forward prediction and data, give more weight to
        # regions where the height map doesn't change very quickly; this value is the coefficient of the exponent of the
        # square TV: weight = exp(-TV_relaxation_coeff * TV ** 2);
        self.use_postpended_channels_for_stitching = False  # this should only be flipped to True by the function,
        # postpend_edge_filtered_channels_for_registration;
        self.gaussian_kernel = None  # define this later if you need it for blurring the background segmentation mask
        # for support constraint regularization;
        self.support_constraint_threshold = None  # define later if  needed;

    def generate_visitation_log(self, restrict_bounds=False, camera_margins=None):
        # specifies which pixels (designated by row/col) from the image stacks imaged which pixels in the recons;
        # this function should be invoked when using the normal mesoSfM functionality after completing 2D registration
        # with camera pose only, likely at a downsampled resolution to save memory;
        # rc_downsamp is a num_images by _ by 2 array, where the spatial dimensions of the image are flattened;
        # restrict_bounds controls whether to deal with situations where coordinates go beyond the edge of the recon
        # field of view;
        # camera_margins: vector of 2 numbers, specifying how much to crop margins from rows and columns,
        # as fractions between 0 and .5;

        assert self.momentum is None and self.batch_size is None  # the visitation log should be generated in one sweep;

        visitation_log_shape = [self.recon_shape[0], self.recon_shape[1], self.num_images]
        # to attain this^ shape, need to create the camera number dimension:

        if camera_margins is None:
            im_downsampled_shape = self.im_downsampled_shape
            rc_downsamp = self.rc_downsamp
        else:
            rc_downsamp = tf.reshape(self.rc_downsamp, (self.num_images,
                                                        self.im_downsampled_shape[0],
                                                        self.im_downsampled_shape[1], 2))
            r_margin = int(camera_margins[0] * self.im_downsampled_shape[0])
            c_margin = int(camera_margins[1] * self.im_downsampled_shape[1])
            im_downsampled_shape = (self.im_downsampled_shape[0] - 2 * r_margin,
                                    self.im_downsampled_shape[1] - 2 * c_margin)
            rc_downsamp = rc_downsamp[:, r_margin:self.im_downsampled_shape[0]-r_margin,
                                         c_margin:self.im_downsampled_shape[1]-c_margin, :]
            rc_downsamp = tf.reshape(rc_downsamp, (self.num_images, -1, 2))

        cam_num = tf.range(self.num_images)  # index for camera number;
        cam_num = tf.broadcast_to(cam_num[:, None], [self.num_images,  # broadcast in prep for concatenation;
                                                     im_downsampled_shape[0] * im_downsampled_shape[1]])
        cam_num = tf.reshape(cam_num, [-1, 1])  # match dims of rc_warp;
        rc_downsamp_flat = tf.reshape(rc_downsamp, [-1, 2])  # also flatten this to match dims of rc_warp;
        rc_downsamp_flat = tf.cast(rc_downsamp_flat, dtype=tf.int32)  # from tf.uint16; for some reason, tf complained;

        # augment the warped coordinates with the camera number:
        # first, retrieve rc_warp from the graph, eagerly:
        rc_warp = self._warp_camera_parameters(tf.cast(rc_downsamp, dtype=self.tf_dtype),
                                               use_radial_deformation=False)
        rc_warp_round = tf.cast(tf.round(rc_warp), dtype=tf.int32)  # obviously need to round to the nearest pixel;
        if restrict_bounds:
            # deal with out-of-bounds coordinates; if this is false and out of boundedness happens, an error wil occur;
            rc_warp_round = self.restrict(rc_warp_round)
        rc_warp_augmented = tf.concat([rc_warp_round, cam_num], axis=1)  # shape: _, 3

        # generate visitation log for row and column separately (non-visited positions in the recon are -1):
        self.visitation_log_r = tf.Variable(np.zeros(visitation_log_shape) - 1, dtype=tf.int32, name='visitation_log_r')
        self.visitation_log_c = tf.Variable(np.zeros(visitation_log_shape) - 1, dtype=tf.int32, name='visitation_log_c')
        self.visitation_log_r.scatter_nd_update(rc_warp_augmented, rc_downsamp_flat[:, 0])
        self.visitation_log_c.scatter_nd_update(rc_warp_augmented, rc_downsamp_flat[:, 1])
        # ^ scatter_nd_update doesn't accumulate values if multiple visits, while scatter_nd does;

        # FINALLY, generate the DENSE (undownsampled) warped coordinates to accompany the dense (undownsampled) image
        # stack; height-map-based coordinate deformations will operate on top of these warped coordinates;
        rc_base = np.transpose(self.rc_base, (0, 2, 1, 3))  # I forget why this is transposed ...;
        self.rc_warp_dense = self._warp_camera_parameters(tf.cast(tf.reshape(rc_base, [self.num_images, -1, 2]),
                                                                  dtype=self.tf_dtype), use_radial_deformation=False)
        self.rc_warp_dense = tf.reshape(self.rc_warp_dense, (self.num_images,
                                                             self.stack.shape[1], self.stack.shape[2], 2))
        self.rc_warp_dense /= self.scale  # remove the scale;

        return self.visitation_log_r, self.visitation_log_c, self.rc_warp_dense, self.scale  # self.stack would make
        # this complete; rc_warp_dense and stack will be used to create the dataset from which batches are generated;
        # self.scale is supplied for convenience to account for the fact that visitation log is downsampled;

    def set_visitation_log(self, visitation_log_vars):
        # generate_visitation_log is supposedly run right after the previous 2D stitching optimization; use the output
        # as the input of this function in the next round of optimization;
        (self.visitation_log_r, self.visitation_log_c,
         self.rc_warp_dense, self.visitation_log_scale) = visitation_log_vars

    def define_network_and_camera_params(self, vanish_point, cam_to_vanish, num_channels_rgb, filters_list=[16]*5,
                                         skip_list=[0]*5, learning_rate=1e-3, architecture='fcnn', optimizer=None,
                                         num_inputs_to_expanded_stack=None
                                         ):
        # specify the fcnn architecture via filters_list and skip_list;
        # camera_params are optimized from a previous round using a flat sample; these are necessary for computing the
        # radial deformation fields given the height map; these should be tf tensors and not np arrays;
        # run this function before running generate_patched_dataset;
        # if optimizer is not None, then override the current optimizer; in this case, learning_rate does nothing;
        # num_inputs_to_expanded_stack: only relevant for fcnn architecture; must agree with expanded stack
        # dims when you call self.expand_stack_channels_for_CNN;

        self.filters_list = filters_list
        self.skip_list = skip_list
        self.architecture = architecture
        self.num_channels_rgb = num_channels_rgb  # if you're passing in a stack that has more channels than rgb because
        # you've stacked neighboring camera views, then this attribute reminds us how many channels the photometric
        # reconstruction will have; if not passing the augmented stack, then nothing will happen, as this will be the
        # same as self.num_channels;

        if architecture == 'fcnn':
            self.num_inputs_to_expanded_stack = num_inputs_to_expanded_stack
            self.network = fcnn(filters_list, skip_list, self.output_nonlinearity,
                                num_inputs=num_inputs_to_expanded_stack)  # build network;
        else:
            raise Exception('invalid network architecture')

        self.optimizer = self.optimizer(learning_rate=learning_rate)
        if optimizer is not None:  # override
            self.optimizer = optimizer

        self.vanish_point = vanish_point  # shape: num_images, 2
        self.cam_to_vanish = cam_to_vanish  # basically camera height; shape: num_images

        self.H_j = self.cam_to_vanish[self.j]  # height of the self.j'th camera; compute this here, because when
        # batching, the jth camera may not be in the batch;

    def _validate_patch_size(self):
        # check if self.patch_size is consistent with length of filters_list/skip_list;
        # this is based on the architecture of fcnn; if it changes, you may need to change this function as well;
        num_block = len(self.filters_list)

        if 'fcnn' in self.architecture:
            subtract = 0  # based on padding behavior;
        else:
            raise Exception('invalid network architecture')

        dim = self.patch_size
        for i in range(num_block):
            dim_ = np.float32((dim - subtract) / 2)
            dim = np.float32((dim - subtract) // 2)
            if ~np.isclose(dim, dim_) or dim <= 0:
                raise Exception('invalid patch size')

    def expand_stack_channels_for_CNN(self, connectivity, camera_array_dims):
        # if you want the CNN to access neighboring camera info;

        self.connectivity = connectivity
        self.camera_array_dims = camera_array_dims
        # reshape flattened camera dimension:
        stack = self.stack.reshape(list(camera_array_dims) + list(self.stack.shape[1:]))
        rc_warp = self.rc_warp_dense.numpy().reshape(list(camera_array_dims) + list(self.rc_warp_dense.shape[1:]))
        new_stack = create_multiview_stack(stack, rc_warp, connectivity)
        self.stack = new_stack.reshape(self.stack.shape[0], *new_stack.shape[2:])
        self.stack_tf = tf.constant(self.stack)  # see constructor on why we need this;
        # update channel info:
        self.num_channels = new_stack.shape[-1]

        # we define the network earlier, which needs to know how many images that have been stacked:
        assert self.num_inputs_to_expanded_stack == self.num_channels // self.num_channels_rgb

    def postpend_edge_filtered_channels_for_registration(self, im_stack, sigmas, gaussian_only=False,
                                                         use_hpf_norm=False):
        # should be run after expand_stack_channels_for_CNN to avoid expanding the postpended channels that this
        # function does;
        # the purpose of this is to add new channels that will be used for registration, but not to be used for the CNN
        # input;
        # sigmas: see edge_filter_rgb_stack;
        # gaussian_only: if True, don't use laplace -- just do Gaussian blur;
        # use_hpf_norm: use a different type of edge filter (see edge_filter_rgb_stack_normalized_hpf);

        self.use_postpended_channels_for_stitching = True  # so that the _backproject_and_predict function knows to use
        # different channels for CNN and registration;

        assert len(sigmas) == self.num_channels_rgb  # for now, this must be True;
        assert im_stack.shape[-1] == self.num_channels_rgb  # we're going to postpend a new stack that has the same
        # number of channels as the original stack, which should be inputted to this function;

        if use_hpf_norm:
            channels_to_postpend = edge_filter_rgb_stack_normalized_hpf(im_stack, sigmas, keep_rgb=False)
        else:
            channels_to_postpend = edge_filter_rgb_stack(im_stack, sigmas, keep_rgb=False, gaussian_only=gaussian_only)

        self.stack = np.concatenate([self.stack, channels_to_postpend], axis=3)  # postpend the new channels;
        self.stack_tf = tf.constant(self.stack)  # see constructor on why we need this;
        # update channel info:
        self.num_channels = self.stack.shape[-1]

    def generate_patched_dataset(self, num_patches, patch_size, patch_recon_size=None, sample_margin=None,
                                 fracture_big_tensors=False, inclusive_patch_selection=False, good_regions=None):
        # generate dataset of patches from the image stack based on a selected patch in the reconstruction;
        # dataset generated from self.visitation_log_r/c, self.rc_warp_dense, self.stack;
        # one element of a batch consists of 2-9 raw image patches (could be more, depending on the
        # mcam configuration) that are known to intersect at a given location in the reconstruction; thus, a batch is a
        # raggedtensor;
        # num_patch is basically the analog of batch_size for mcam3d;
        # run this function after define_network_and_camera_parameters;
        # patch_recon_size is the size of the tensor you're scatter_nd'ing the patches into; if not supplied, it will
        # default to patch_size*3;
        # sample_margin: how much along the border of the reconstruction to exclude from sampling; array of 4 numbers,
        # specifying r0, r1, c1, c0 (each between 0 and 0.5);
        # fracture_big_tensors: dataset complains if stack and rc_warp_dense are too big (such as for 54 3000x4000 MCAM
        # datasets);
        # inclusive_patch_selection: when retrieving from the visitation log, check intersection with any point within
        # r:r+patch_size, c:c+patch_size (i.e., old behavior); otherwise, only check r,c;
        # good regions: if not None, then when generating random coordinates, only accept if they are True/1 in the
        # good_regions array, which is of the same dimensions and scale as the visitation log; boolean array;

        self.num_patches = num_patches
        self.patch_size = patch_size
        assert patch_size % 2 == 0  # we'll be dividing this by 2 in _gather_image_patches;
        self._validate_patch_size()
        if patch_recon_size is None:
            self.patch_recon_size = 3 * self.patch_size
        else:
            self.patch_recon_size = patch_recon_size

        if fracture_big_tensors:
            self.fracture_size = 1
            self.stack_tf0 = self.stack_tf[0:1]
            self.rc_warp_dense0 = self.rc_warp_dense[0:1]
            self.stack_tf1 = self.stack_tf[1:2]
            self.rc_warp_dense1 = self.rc_warp_dense[1:2]
            self.stack_tf2 = self.stack_tf[2:3]
            self.rc_warp_dense2 = self.rc_warp_dense[2:3]
            self.stack_tf3 = self.stack_tf[3:4]
            self.rc_warp_dense3 = self.rc_warp_dense[3:4]
            self.stack_tf4 = self.stack_tf[4:5]
            self.rc_warp_dense4 = self.rc_warp_dense[4:5]
            self.stack_tf5 = self.stack_tf[5:6]
            self.rc_warp_dense5 = self.rc_warp_dense[5:6]
            self.stack_tf6 = self.stack_tf[6:7]
            self.rc_warp_dense6 = self.rc_warp_dense[6:7]
            self.stack_tf7 = self.stack_tf[7:8]
            self.rc_warp_dense7 = self.rc_warp_dense[7:8]
            self.stack_tf8 = self.stack_tf[8:9]
            self.rc_warp_dense8 = self.rc_warp_dense[8:9]
            self.stack_tf9 = self.stack_tf[9:10]
            self.rc_warp_dense9 = self.rc_warp_dense[9:10]
            self.stack_tf10 = self.stack_tf[10:11]
            self.rc_warp_dense10 = self.rc_warp_dense[10:11]
            self.stack_tf11 = self.stack_tf[11:12]
            self.rc_warp_dense11 = self.rc_warp_dense[11:12]
            self.stack_tf12 = self.stack_tf[12:13]
            self.rc_warp_dense12 = self.rc_warp_dense[12:13]
            self.stack_tf13 = self.stack_tf[13:14]
            self.rc_warp_dense13 = self.rc_warp_dense[13:14]
            self.stack_tf14 = self.stack_tf[14:15]
            self.rc_warp_dense14 = self.rc_warp_dense[14:15]
            self.stack_tf15 = self.stack_tf[15:16]
            self.rc_warp_dense15 = self.rc_warp_dense[15:16]
            self.stack_tf16 = self.stack_tf[16:17]
            self.rc_warp_dense16 = self.rc_warp_dense[16:17]
            self.stack_tf17 = self.stack_tf[17:18]
            self.rc_warp_dense17 = self.rc_warp_dense[17:18]
            self.stack_tf18 = self.stack_tf[18:19]
            self.rc_warp_dense18 = self.rc_warp_dense[18:19]
            self.stack_tf19 = self.stack_tf[19:20]
            self.rc_warp_dense19 = self.rc_warp_dense[19:20]
            self.stack_tf20 = self.stack_tf[20:21]
            self.rc_warp_dense20 = self.rc_warp_dense[20:21]
            self.stack_tf21 = self.stack_tf[21:22]
            self.rc_warp_dense21 = self.rc_warp_dense[21:22]
            self.stack_tf22 = self.stack_tf[22:23]
            self.rc_warp_dense22 = self.rc_warp_dense[22:23]
            self.stack_tf23 = self.stack_tf[23:24]
            self.rc_warp_dense23 = self.rc_warp_dense[23:24]
            self.stack_tf24 = self.stack_tf[24:25]
            self.rc_warp_dense24 = self.rc_warp_dense[24:25]
            self.stack_tf25 = self.stack_tf[25:26]
            self.rc_warp_dense25 = self.rc_warp_dense[25:26]
            self.stack_tf26 = self.stack_tf[26:27]
            self.rc_warp_dense26 = self.rc_warp_dense[26:27]
            self.stack_tf27 = self.stack_tf[27:28]
            self.rc_warp_dense27 = self.rc_warp_dense[27:28]
            self.stack_tf28 = self.stack_tf[28:29]
            self.rc_warp_dense28 = self.rc_warp_dense[28:29]
            self.stack_tf29 = self.stack_tf[29:30]
            self.rc_warp_dense29 = self.rc_warp_dense[29:30]
            self.stack_tf30 = self.stack_tf[30:31]
            self.rc_warp_dense30 = self.rc_warp_dense[30:31]
            self.stack_tf31 = self.stack_tf[31:32]
            self.rc_warp_dense31 = self.rc_warp_dense[31:32]
            self.stack_tf32 = self.stack_tf[32:33]
            self.rc_warp_dense32 = self.rc_warp_dense[32:33]
            self.stack_tf33 = self.stack_tf[33:34]
            self.rc_warp_dense33 = self.rc_warp_dense[33:34]
            self.stack_tf34 = self.stack_tf[34:35]
            self.rc_warp_dense34 = self.rc_warp_dense[34:35]
            self.stack_tf35 = self.stack_tf[35:36]
            self.rc_warp_dense35 = self.rc_warp_dense[35:36]
            self.stack_tf36 = self.stack_tf[36:37]
            self.rc_warp_dense36 = self.rc_warp_dense[36:37]
            self.stack_tf37 = self.stack_tf[37:38]
            self.rc_warp_dense37 = self.rc_warp_dense[37:38]
            self.stack_tf38 = self.stack_tf[38:39]
            self.rc_warp_dense38 = self.rc_warp_dense[38:39]
            self.stack_tf39 = self.stack_tf[39:40]
            self.rc_warp_dense39 = self.rc_warp_dense[39:40]
            self.stack_tf40 = self.stack_tf[40:41]
            self.rc_warp_dense40 = self.rc_warp_dense[40:41]
            self.stack_tf41 = self.stack_tf[41:42]
            self.rc_warp_dense41 = self.rc_warp_dense[41:42]
            self.stack_tf42 = self.stack_tf[42:43]
            self.rc_warp_dense42 = self.rc_warp_dense[42:43]
            self.stack_tf43 = self.stack_tf[43:44]
            self.rc_warp_dense43 = self.rc_warp_dense[43:44]
            self.stack_tf44 = self.stack_tf[44:45]
            self.rc_warp_dense44 = self.rc_warp_dense[44:45]
            self.stack_tf45 = self.stack_tf[45:46]
            self.rc_warp_dense45 = self.rc_warp_dense[45:46]
            self.stack_tf46 = self.stack_tf[46:47]
            self.rc_warp_dense46 = self.rc_warp_dense[46:47]
            self.stack_tf47 = self.stack_tf[47:48]
            self.rc_warp_dense47 = self.rc_warp_dense[47:48]
            self.stack_tf48 = self.stack_tf[48:49]
            self.rc_warp_dense48 = self.rc_warp_dense[48:49]
            self.stack_tf49 = self.stack_tf[49:50]
            self.rc_warp_dense49 = self.rc_warp_dense[49:50]
            self.stack_tf50 = self.stack_tf[50:51]
            self.rc_warp_dense50 = self.rc_warp_dense[50:51]
            self.stack_tf51 = self.stack_tf[51:52]
            self.rc_warp_dense51 = self.rc_warp_dense[51:52]
            self.stack_tf52 = self.stack_tf[52:53]
            self.rc_warp_dense52 = self.rc_warp_dense[52:53]
            self.stack_tf53 = self.stack_tf[53:54]
            self.rc_warp_dense53 = self.rc_warp_dense[53:54]
        else:
            self.fracture_size = None

        # run the network once so that we can access network.trainable_variables
        if self.use_postpended_channels_for_stitching:
            num_channels = self.num_channels - self.num_channels_rgb  # self.num_channels_rgb channels postpended, which
            # are for registration, not CNN input;
        else:
            num_channels = self.num_channels

        out = self.network(tf.zeros([1, self.patch_size, self.patch_size, num_channels], dtype=self.tf_dtype))
        self.output_patch_size = out.numpy().shape[1]  # might not be same as self.patch_size if not using padded convs;
        print('Output patch size: ' + str(self.output_patch_size))
        if self.patch_size != self.output_patch_size:
            print('Warning: training will work with output != input size, but inference on full images will not')

        # ignore regions beyond the visitation log:
        log_nonzero = (self.visitation_log_r.numpy() > 0).sum(2)
        r_nonzero, c_nonzero = np.nonzero(log_nonzero)
        r0 = r_nonzero.min() / self.visitation_log_scale
        r1 = r_nonzero.max() / self.visitation_log_scale
        c0 = c_nonzero.min() / self.visitation_log_scale
        c1 = c_nonzero.max() / self.visitation_log_scale
        r_width = r1 - r0
        c_width = c1 - c0

        if sample_margin is not None:
            row_low = r0 + sample_margin[0] * r_width
            row_high = r0 + (1 - sample_margin[1]) * r_width - self.patch_size - 1
            col_low = c0 + sample_margin[2] * c_width
            col_high = c0 + (1 - sample_margin[3]) * c_width - self.patch_size - 1
        else:
            row_low = r0
            row_high = r1 - self.patch_size - 1
            col_low = c0
            col_high = c1 - self.patch_size - 1

        # tf complains if this lambda function isn't defined on a standalone line:
        # (recon_shape_base is the non-downsampled size);
        if good_regions is None:
            generate_rand_coord = lambda x: (tf.random.uniform((), row_low, row_high),
                                             tf.random.uniform((), col_low, col_high))
        else:
            def generate_rand_coord(x):
                # keep generating random uniform coordinates until they land in a good region;
                def body(x=0, y=0):
                    # these coordinates are not scaled by self.visitation_log_scale
                    row_ = tf.random.uniform((), row_low, row_high)
                    col_ = tf.random.uniform((), col_low, col_high)

                    return row_, col_

                def cond(row_, col_):
                    # tests if good region
                    row_scale = tf.cast(row_ * self.visitation_log_scale, dtype=tf.int32)
                    col_scale = tf.cast(col_ * self.visitation_log_scale, dtype=tf.int32)
                    patch_size_scale = np.int32(patch_size * self.visitation_log_scale)

                    patch = good_regions[row_scale:row_scale + patch_size_scale,
                                         col_scale:col_scale+ + patch_size_scale]
                    frac_good = tf.reduce_mean(tf.cast(patch, dtype=tf.float32))
                    return frac_good < .1  # while this is true, keep generating

                row, col = body()
                row, col = tf.while_loop(cond=cond, body=body, loop_vars=(row, col))

                return row, col

        gather_image_patches = lambda r, c: self._gather_image_patches(r, c, inclusive_patch_selection)
        if inclusive_patch_selection:
            print('WARNING: inclusive_patch_selection behavior has changed and not been tested!')

        dataset = (tf.data.Dataset.range(1)  # dummy dataset;
                   .map(generate_rand_coord)  # generate one random coordinate;
                   .map(gather_image_patches)
                   .repeat(None)  # generate infinite number of patches;
                   .apply(tf.data.experimental.dense_to_ragged_batch(batch_size=self.num_patches))  # different number
                   # of image patches per reconstruction patch;
                   .prefetch(1)
                   )  # ragged function seems to only work in tf2.3 (2.2 fails);
        return dataset

    def _gather_image_patches(self, r, c, inclusive_patch_selection):
        # used by generate_patched_dataset, but can also be used by user in eager mode for diagnostics;
        # given r(ow) and c(olumn), corresponding to upper left corner, identify the image patches that overlap, based
        # on visitation_log; return patches from the raw image stack along with patches from the corresponding
        # rc_warp_dense coordinates;
        r = tf.cast(r * self.visitation_log_scale, dtype=tf.int32)
        c = tf.cast(c * self.visitation_log_scale, dtype=tf.int32)

        # retrieve records from visitation log:
        # remember that the visitation log has -1 for unvisited pixels!
        patch_size_scaled = tf.cast(self.patch_size * self.visitation_log_scale, dtype=tf.int32)
        if inclusive_patch_selection:
            retrieved_record_r = self.visitation_log_r[r:r + patch_size_scaled, c:c + patch_size_scaled]
            retrieved_record_c = self.visitation_log_c[r:r + patch_size_scaled, c:c + patch_size_scaled]
            # shapes: (patch_size, patch_size, num_images);
        else:
            r += patch_size_scaled // 2  # center the coordinate; sampling is based on upper left position;
            c += patch_size_scaled // 2
            retrieved_record_r = self.visitation_log_r[r:r + 1, c:c + 1]
            retrieved_record_c = self.visitation_log_c[r:r + 1, c:c + 1]

        max_r = tf.reduce_max(retrieved_record_r, axis=(0, 1))  # shape: num_images;
        max_c = tf.reduce_max(retrieved_record_c, axis=(0, 1))

        # if max_r/c is less than the max dim of images and greater than patch_size, then crop image from max-patch_size
        # to max; if max is less than patch_size, then crop image from 0 to patch_size; if max is greater than
        # image size, then crop from image_size-patch size to image_size;

        # first, filter images by those which visited the current patch (if unvisited, max_r will be -1);
        inds_images_to_use = tf.cast(tf.where(max_r >= 0)[:, 0], tf.int32)  # length < num_images;

        # I think I have to use a for-loop here, at least to avoid doing a messy tf.gather operation;
        im_patches = tf.TensorArray(tf.uint8, size=len(inds_images_to_use),  # use in lieu of list;
                                    element_shape=(self.patch_size, self.patch_size, self.num_channels))
        rc_warp_patches = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                         element_shape=(self.patch_size, self.patch_size, 2))
        cam_to_vanish_batch = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                             element_shape=())
        vanish_point_batch = tf.TensorArray(tf.float32, size=len(inds_images_to_use),
                                            element_shape=(2,))
        inds_images_to_use_ = tf.TensorArray(tf.int32, size=len(inds_images_to_use),
                                         element_shape=())  # this may seem useless, but for some reason tf doesn't
        # combine inds_images_to_use into a ragged batch (at least in tf2.3); tf.zeros_like(inds_images_to_use) also
        # fails, but tf.zeros(len(inds_images_to_use)) succeeds;

        for i in tf.range(len(inds_images_to_use)):  # can't use enumerate, or else tf might interpret as a python loop;
            ind = inds_images_to_use[i]
            max_r_ind = max_r[ind]  # one number;
            max_c_ind = max_c[ind]  # one number;

            # three cases:
            # 1) max_r_ind < self.patch_size / 2
            # 2) max_r_ind >= self.stack.shape[1] - self.patch_size / 2
            # 3) self.patch_size <= max_r_ind < self.stack.shape[1] (treated as default below)
            # (and the same one for c)
            # note that patch_size is even
            r_start, r_end = tf.case([(tf.less(max_r_ind, self.patch_size // 2), lambda: (0, self.patch_size)),
                                      (tf.greater_equal(max_r_ind, self.stack.shape[1] - self.patch_size // 2),
                                       lambda: (self.stack.shape[1] - self.patch_size, self.stack.shape[1])),
                                      ], default=lambda: (max_r_ind - self.patch_size // 2,
                                                          max_r_ind + self.patch_size // 2))

            c_start, c_end = tf.case([(tf.less(max_c_ind, self.patch_size // 2), lambda: (0, self.patch_size)),
                                      (tf.greater_equal(max_c_ind, self.stack.shape[2] - self.patch_size // 2),
                                       lambda: (self.stack.shape[2] - self.patch_size, self.stack.shape[2])),
                                      ], default=lambda: (max_c_ind - self.patch_size // 2,
                                                          max_c_ind + self.patch_size // 2))

            if self.fracture_size is not None:
                fracture_num = tf.cast(ind / self.fracture_size, dtype=tf.int32)  # which fracture?
                ind_fracture = tf.math.floormod(ind, self.fracture_size)  # within that fracture, which index?
                stack_tf, rc_warp_dense = tf.switch_case(fracture_num, [lambda: (self.stack_tf0, self.rc_warp_dense0),
                                                                        lambda: (self.stack_tf1, self.rc_warp_dense1),
                                                                        lambda: (self.stack_tf2, self.rc_warp_dense2),
                                                                        lambda: (self.stack_tf3, self.rc_warp_dense3),
                                                                        lambda: (self.stack_tf4, self.rc_warp_dense4),
                                                                        lambda: (self.stack_tf5, self.rc_warp_dense5),
                                                                        lambda: (self.stack_tf6, self.rc_warp_dense6),
                                                                        lambda: (self.stack_tf7, self.rc_warp_dense7),
                                                                        lambda: (self.stack_tf8, self.rc_warp_dense8),
                                                                        lambda: (self.stack_tf9, self.rc_warp_dense9),
                                                                        lambda: (self.stack_tf10, self.rc_warp_dense10),
                                                                        lambda: (self.stack_tf11, self.rc_warp_dense11),
                                                                        lambda: (self.stack_tf12, self.rc_warp_dense12),
                                                                        lambda: (self.stack_tf13, self.rc_warp_dense13),
                                                                        lambda: (self.stack_tf14, self.rc_warp_dense14),
                                                                        lambda: (self.stack_tf15, self.rc_warp_dense15),
                                                                        lambda: (self.stack_tf16, self.rc_warp_dense16),
                                                                        lambda: (self.stack_tf17, self.rc_warp_dense17),
                                                                        lambda: (self.stack_tf18, self.rc_warp_dense18),
                                                                        lambda: (self.stack_tf19, self.rc_warp_dense19),
                                                                        lambda: (self.stack_tf20, self.rc_warp_dense20),
                                                                        lambda: (self.stack_tf21, self.rc_warp_dense21),
                                                                        lambda: (self.stack_tf22, self.rc_warp_dense22),
                                                                        lambda: (self.stack_tf23, self.rc_warp_dense23),
                                                                        lambda: (self.stack_tf24, self.rc_warp_dense24),
                                                                        lambda: (self.stack_tf25, self.rc_warp_dense25),
                                                                        lambda: (self.stack_tf26, self.rc_warp_dense26),
                                                                        lambda: (self.stack_tf27, self.rc_warp_dense27),
                                                                        lambda: (self.stack_tf28, self.rc_warp_dense28),
                                                                        lambda: (self.stack_tf29, self.rc_warp_dense29),
                                                                        lambda: (self.stack_tf30, self.rc_warp_dense30),
                                                                        lambda: (self.stack_tf31, self.rc_warp_dense31),
                                                                        lambda: (self.stack_tf32, self.rc_warp_dense32),
                                                                        lambda: (self.stack_tf33, self.rc_warp_dense33),
                                                                        lambda: (self.stack_tf34, self.rc_warp_dense34),
                                                                        lambda: (self.stack_tf35, self.rc_warp_dense35),
                                                                        lambda: (self.stack_tf36, self.rc_warp_dense36),
                                                                        lambda: (self.stack_tf37, self.rc_warp_dense37),
                                                                        lambda: (self.stack_tf38, self.rc_warp_dense38),
                                                                        lambda: (self.stack_tf39, self.rc_warp_dense39),
                                                                        lambda: (self.stack_tf40, self.rc_warp_dense40),
                                                                        lambda: (self.stack_tf41, self.rc_warp_dense41),
                                                                        lambda: (self.stack_tf42, self.rc_warp_dense42),
                                                                        lambda: (self.stack_tf43, self.rc_warp_dense43),
                                                                        lambda: (self.stack_tf44, self.rc_warp_dense44),
                                                                        lambda: (self.stack_tf45, self.rc_warp_dense45),
                                                                        lambda: (self.stack_tf46, self.rc_warp_dense46),
                                                                        lambda: (self.stack_tf47, self.rc_warp_dense47),
                                                                        lambda: (self.stack_tf48, self.rc_warp_dense48),
                                                                        lambda: (self.stack_tf49, self.rc_warp_dense49),
                                                                        lambda: (self.stack_tf50, self.rc_warp_dense50),
                                                                        lambda: (self.stack_tf51, self.rc_warp_dense51),
                                                                        lambda: (self.stack_tf52, self.rc_warp_dense52),
                                                                        lambda: (self.stack_tf53, self.rc_warp_dense53),
                                                                        ])
                rc_warp_patches = rc_warp_patches.write(i, rc_warp_dense[ind_fracture, r_start:r_end, c_start:c_end, :])
                im_patches = im_patches.write(i, stack_tf[ind_fracture, r_start:r_end, c_start:c_end, :])
            else:
                # no fracturing; index into the whole tensor:
                rc_warp_patches = rc_warp_patches.write(i, self.rc_warp_dense[ind, r_start:r_end, c_start:c_end, :])
                im_patches = im_patches.write(i, self.stack_tf[ind, r_start:r_end, c_start:c_end, :])
                # tf complains if I use self.stack (numpy version) rather than self.stack_tf (tf version)^;

            # these variables are small; no need to give the fracture treatment;
            inds_images_to_use_ = inds_images_to_use_.write(i, ind)
            vanish_point_batch = vanish_point_batch.write(i, self.vanish_point[ind])
            cam_to_vanish_batch = cam_to_vanish_batch.write(i, self.cam_to_vanish[ind])

        return (im_patches.stack(),  # shape: _ by patch_size by patch_size by 3;
                rc_warp_patches.stack(),  # shape: _ by patch_size by patch_size by 2;
                vanish_point_batch.stack(),  # shape: _ by 2;
                cam_to_vanish_batch.stack(),  # shape: _
                inds_images_to_use_.stack(),  # shape: _
                r, c)  # also return the random coordinate;

    def _backproject_and_predict(self, im_patches, rc_warp_patches, vanish_point_batch, cam_to_vanish_batch,
                                 inds_images_to_use, r, c, stop_gradient=True, dither_coords=False,
                                 downsample_factor=1, support_constraint_threshold=None):
        # generate camera-centric height map, then warp the coordinates, then backproject to get construction;
        # input arguments are from tf.dataset;
        # specifically, unpack the ragged batches (effectively flattening along the ragged dimension) and use row_splits
        # to keep track of the batch boundaries (or better yet, value_rowids(), which gives me the indices for batch
        # membrship, which I can use for scatter_nd);
        # the constant scale factors used in mesoSfM are not used here for simplicity;
        # use_postpended_channels_for_stitching: if True, then use num_channels_rgb channels from the bottom of channels
        # stack to be used for registration instead of the channels that are inputted to the CNN (the first few
        # channels);
        # to be clear, num_channels is always the number of channels in the stack, including CNN input, augmented
        # channels from neighboring cameras, and additional post-pended channels; num_channels_rgb is always the number
        # of channels used for registration; for simplicity, number of channels inputted to CNN is also the number of
        # channels used for registration;
        # downsample_factor: downsample the patched reconstruction (for multi-resolution optimization);

        patch_recon_size = tf.cast(self.patch_recon_size / downsample_factor, dtype=tf.int32)

        # unpack batch:
        im_flat = tf.cast(im_patches.values, self.tf_dtype)  # flattens ragged dimension;
        # new shape^: _, patch, patch, channels;
        partitions = tf.cast(im_patches.value_rowids(), tf.int32)  # shape: _;
        rc_warp_flat = rc_warp_patches.values  # shape: _, patch, patch, 2;
        vanish_point_flat = vanish_point_batch.values  # shape: _, 2;
        cam_to_vanish_flat = cam_to_vanish_batch.values  # shape: _;

        # generate height map:
        if self.recompute_CNN:
            network = tf.recompute_grad(self.network)
        else:
            network = self.network

        if self.use_postpended_channels_for_stitching:
            CNN_input = im_flat[..., :-self.num_channels_rgb]  # last few channels are for stitching only, not for CNN!
        else:
            CNN_input = im_flat

        fcnn_out = network(CNN_input)

        # convert im_flat to what you need for registration, since CNN has already been used;
        if self.use_postpended_channels_for_stitching:
            im_flat = im_flat[..., -self.num_channels_rgb:]  # only need last few channels for stitching;
        else:
            if self.num_channels != self.num_channels_rgb:  # i.e., you've augmented im_stack with more channels for CNN
                im_flat = im_flat[..., :self.num_channels_rgb]  # we've used the CNN; don't need the extra channels;

        ego_height = tf.reduce_mean(fcnn_out, [-1]) * self.unet_scale  # remove feature dimension;

        if support_constraint_threshold is not None:
            self.support_constraint_threshold = support_constraint_threshold  # to use in generate_full_recon;
            im_green = CNN_input[..., 1]  # shape: _, patch, patch
            bkgd_mask = tf.cast(im_green > support_constraint_threshold, dtype=tf.float32)
            if self.gaussian_kernel is not None:
                bkgd_mask = tf.nn.conv2d(bkgd_mask[:, :, :, None], self.gaussian_kernel[:, :, None, None],
                                         strides=1, padding='SAME')[:, :, :, 0]  # blur mask to reduce sharp edges;
            self.support_loss = tf.reduce_mean(bkgd_mask * ego_height ** 2)  # things segmented out as background should
            # ... be forced to be close to 0;

        # flatten out spatial dims:
        if self.patch_size > self.output_patch_size:
            # need to crop the patches from the input
            margin = (self.patch_size - self.output_patch_size) // 2
            ego_height = tf.reshape(ego_height, [-1, self.output_patch_size ** 2])
            rc_warp = tf.reshape(rc_warp_flat[:, margin:-margin, margin:-margin, :],
                                 [-1, self.output_patch_size ** 2, 2])
            im_flat = tf.reshape(im_flat[:, margin:-margin, margin:-margin, :],
                                 [-1, self.output_patch_size ** 2, self.num_channels_rgb])

        elif self.patch_size < self.output_patch_size:
            # this means you have more upsampling layers than downsampling ...
            raise Exception('output patch size > input patch size')
        else:  # output size = input size;
            ego_height = tf.reshape(ego_height, [-1, self.patch_size ** 2])
            rc_warp = tf.reshape(rc_warp_flat, [-1, self.patch_size ** 2, 2])
            im_flat = tf.reshape(im_flat, [-1, self.patch_size ** 2, self.num_channels_rgb])

        H = cam_to_vanish_flat[:, None]  # camera heights
        M_j = self.magnification_j
        f_eff = self.effective_focal_length_mm

        # warp the rc_warp further using height map;
        r = rc_warp - vanish_point_flat[:, None, :]  # lateral distance to vanishing point;
        delta_radial = ego_height / f_eff / (1 + 1 / M_j * H / self.H_j)  # radial deform field based on height map;
        rc = r * (1 - delta_radial[:, :, None]) + vanish_point_flat[:, None, :]  # add back vanishing point;
        # shape of rc: _, patch**2, 2;

        # stacking:

        self.im = tf.concat([im_flat, ego_height[:, :, None]], axis=2)  # add height as 4th channel;

        if dither_coords:
            # random rotation:
            theta = tf.random.uniform((), 0, 2 * np.pi, dtype=self.tf_dtype)
            cos = tf.cos(theta)
            sin = tf.sin(theta)
            rotmat = tf.stack([[cos, sin], [-sin, cos]])
            rc = tf.einsum('abc,cd->abd', rc, rotmat)
            # random anisotropic scaling:
            # rc = rc * tf.random.uniform([1, 1, 2], .5, 1.5, dtype=self.tf_dtype)
            # random sub-pixel translation:
            rc = rc + tf.random.uniform([1, 1, 2], -1, 1, dtype=self.tf_dtype)

        rc = rc / downsample_factor

        # backprojection coordinate generation, as usual:
        # neighboring pixels:
        rc_floor = tf.floor(rc)
        rc_ceil = rc_floor + 1

        # distance to neighboring pixels:
        frc = rc - rc_floor
        crc = rc_ceil - rc

        # cast
        rc_floor = tf.cast(rc_floor, tf.int32)
        rc_ceil = tf.cast(rc_ceil, tf.int32)

        # force the use of mod as the restrict function for dealing with out-of-bounds coordinates; this means that if
        # you make the patch recon large, the code will be tolerant to errors in centering the patches;
        restrict = lambda x: tf.math.floormod(x, patch_recon_size)

        self.rc_ff = restrict(rc_floor)
        self.rc_cc = restrict(rc_ceil)
        self.rc_cf = restrict(tf.stack([rc_ceil[:, :, 0], rc_floor[:, :, 1]], 2))
        self.rc_fc = restrict(tf.stack([rc_floor[:, :, 0], rc_ceil[:, :, 1]], 2))

        self.frc = tf.exp(-frc ** 2 / 2. / self.sig_proj ** 2)
        self.crc = tf.exp(-crc ** 2 / 2. / self.sig_proj ** 2)  # shape: _, patch**2, 2

        # augmented coordinates:
        rc_4 = tf.stack([self.rc_ff, self.rc_cc, self.rc_cf, self.rc_fc], 0)  # shape: 4, _, patch**2, 2;
        rcp_4 = tf.concat([rc_4, tf.broadcast_to(partitions[None, :, None, None],  # shape: 4, _, patch**2, 3;
                                                (4, len(partitions), self.output_patch_size ** 2, 1))], 3)
        rcp_4 = tf.reshape(rcp_4, [-1, 3])  # finally, flatten;

        # interpolated:
        im_4 = tf.stack([self.im * self.frc[:, :, 0, None] * self.frc[:, :, 1, None],
                         self.im * self.crc[:, :, 0, None] * self.crc[:, :, 1, None],
                         self.im * self.crc[:, :, 0, None] * self.frc[:, :, 1, None], # shape: 4, _, patch**2, channels;
                         self.im * self.frc[:, :, 0, None] * self.crc[:, :, 1, None]], 0)
        w_4 = tf.stack([self.frc[:, :, 0] * self.frc[:, :, 1],
                        self.crc[:, :, 0] * self.crc[:, :, 1],
                        self.crc[:, :, 0] * self.frc[:, :, 1],
                        self.frc[:, :, 0] * self.crc[:, :, 1]], 0)  # shape: 4, _, patch**2;
        im_4 = tf.reshape(im_4, [-1, self.num_channels_rgb + 1])
        w_4= tf.reshape(w_4,[-1])

        # backproject:
        self.normalize = tf.scatter_nd(rcp_4, w_4, [patch_recon_size, patch_recon_size, self.num_patches])
        self.recon = tf.scatter_nd(rcp_4, im_4, [patch_recon_size, patch_recon_size,
                                                 self.num_patches, self.num_channels_rgb + 1])
        self.recon = tf.math.divide_no_nan(self.recon, self.normalize[:, :, :, None])
        # shape: patch_recon_size, patch_recon_size, num patches, num channels;

        if stop_gradient:
            self.recon = tf.stop_gradient(self.recon)

        # now, forward prediction:
        gathered = tf.gather_nd(self.recon, rcp_4)  # shape: 4*_*patch*patch, channels;
        gathered = tf.reshape(gathered, (4, -1, self.output_patch_size ** 2, self.num_channels_rgb + 1))
        ff, cc, cf, fc = tf.unstack(gathered, num=4, axis=0)  # shape of each: _, patch*patch, channels;

        self.forward = (ff * self.frc[:, :, 0, None] * self.frc[:, :, 1, None] +
                        cc * self.crc[:, :, 0, None] * self.crc[:, :, 1, None] +
                        cf * self.crc[:, :, 0, None] * self.frc[:, :, 1, None] +
                        fc * self.frc[:, :, 0, None] * self.crc[:, :, 1, None])

        self.forward /= ((self.frc[:, :, 0, None] * self.frc[:, :, 1, None]) +
                         (self.crc[:, :, 0, None] * self.crc[:, :, 1, None]) +
                         (self.crc[:, :, 0, None] * self.frc[:, :, 1, None]) +
                         (self.frc[:, :, 0, None] * self.crc[:, :, 1, None]))  # shape: _, patch**2, channels;

        # error between prediction and data:
        # split off the last dimension, the height dimension, to compute the height map MSE:
        self.forward_height = self.forward[:, :, -1]
        self.error_height = self.forward_height - self.im[:, :, -1]
        self.error = self.forward[:, :, :-1] - self.im[:, :, :-1]  # remaining channels are the actual recon;

        if self.TV_relaxation_coeff is None:
            self.MSE_height = tf.reduce_mean(self.error_height ** 2)
            self.MSE = tf.reduce_mean(self.error ** 2)
            self.loss_weight = None
        else:
            height = self.recon[:, :, :, -1]
            d0 = height[1:, :-1] - height[:-1, :-1]
            d1 = height[:-1, 1:] - height[:-1, :-1]
            self.TV2 = d0 ** 2 + d1 ** 2
            self.TV2 = tf.stop_gradient(self.TV2)
            loss_weight = tf.gather_nd(self.TV2, rcp_4)
            self.loss_weight = tf.reshape(loss_weight,
                                          (4, -1, self.output_patch_size ** 2))[0]  # pick one of the 4 pixels;
            self.loss_weight = tf.exp(-self.TV_relaxation_coeff * self.loss_weight)
            self.MSE = tf.reduce_mean(self.loss_weight[:, :, None] * self.error ** 2)
            self.MSE_height = tf.reduce_mean(self.loss_weight * self.error_height ** 2)
            self.tensors_to_track['loss_weight'] = self.loss_weight

        return self.recon, self.normalize, self.forward, self.loss_weight

    @tf.function
    def gradient_update_patch(self, batch, height_map_reg_coef=None, return_tracked_tensors=False, stop_gradient=True,
                              return_loss_only=False, return_gradients=False, clip_gradient_norm=None,
                              dither_coords=False, downsample_factor=1,
                              support_constraint_coef=None, support_constraint_threshold=None):
        # clip_gradient_norm: pick a threshold to clip to (tf.clip_by_norm);
        # support constraint threshold and support constraint coef must both defined if you want to use support
        # constraint as a regularization term; the threshold is based on the green channel -- anything in the green
        # channel less than this threshold is considered the object; an L2 loss is performed on pixels greater than
        # this threshold; note that the background height will be regularized to 0;

        with tf.GradientTape() as tape:
            self._backproject_and_predict(*batch, stop_gradient, dither_coords, downsample_factor,
                                          support_constraint_threshold)

            loss_list = [self.MSE]
            if height_map_reg_coef is not None:
                loss_list.append(height_map_reg_coef * self.MSE_height)
            if support_constraint_coef is not None and support_constraint_threshold is not None:
                loss_list.append(support_constraint_coef * self.support_loss)

            loss = tf.reduce_sum(loss_list)

        grads = tape.gradient(loss, self.network.trainable_variables)
        if clip_gradient_norm is not None:
            grads, global_norm = tf.clip_by_global_norm(grads, clip_gradient_norm)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))

        if return_loss_only:
            return_list = [loss_list]
        else:
            if return_tracked_tensors:
                return_list = [loss_list, self.recon, self.normalize, self.tensors_to_track]
            else:
                return_list = [loss_list, self.recon, self.normalize]

        if return_gradients:
            return_list.append(grads)
        if clip_gradient_norm is not None:
            return_list.append(global_norm)

        return return_list

    def generate_full_recon(self, margin=None, stitch_rgb=True):
        # run the network on each image in a for loop, and backproject sequentially;
        # may be a good idea to do this on CPU for now;
        # margin is a value in pixels that specifies how much to crop the output of the CNN to remove edge effects;
        # stitch_rgb: only relevant when self.use_postpended_channels_for_stitching is True; instead of reconstructing
        # with the postpended channels, as is done during training, reconstruct using the rgb channels, since we just
        # want a nice forward prediction;

        # accumulate these tensors with the for loop:
        recon_cumulative = tf.zeros(list(self.recon_shape_base) + [self.num_channels_rgb + 1], dtype=self.tf_dtype)
        normalize_cumulative = tf.zeros(self.recon_shape_base, dtype=self.tf_dtype)

        # create padding and depadding layers (this should only ever be used for full reconstruction generation):
        self.padded_shape = [self.network.get_compatible_size(dim) for dim in self.stack.shape[1:3]]
        pad_r = self.padded_shape[0] - self.stack.shape[1]
        pad_c = self.padded_shape[1] - self.stack.shape[2]
        pad_top = pad_r // 2
        pad_bottom = int(tf.math.ceil(pad_r / 2))
        pad_left = pad_c // 2
        pad_right = int(tf.math.ceil(pad_c / 2))
        pad_specs = ((pad_top, pad_bottom), (pad_left, pad_right))
        pad_layer = tf.keras.layers.ZeroPadding2D(pad_specs)
        depad_layer = tf.keras.layers.Cropping2D(pad_specs)

        restrict = lambda x: tf.math.floormod(x, self.recon_shape_base)

        for i, (im, rc_warp, vanish_point, cam_to_vanish) in tqdm(enumerate(zip(self.stack, self.rc_warp_dense,
                                                                                self.vanish_point, self.cam_to_vanish)),
                                                                  total=len(self.stack)):
            # im shape: row, col, num_dim;
            # rc_warp shape: 1row, col, 2;
            # vanish_point shape: 2;
            # cam_to_vanish shape: ();

            im = tf.cast(im, dtype=self.tf_dtype)[None]  # cast from uint8 to float32; add batch dim;

            if self.use_postpended_channels_for_stitching:
                CNN_input = im[..., :-self.num_channels_rgb]  # last few channels are for stitching only, not for CNN!
            else:
                CNN_input = im

            # generate height map:
            im_pad = pad_layer(CNN_input)  # pad to a shape the network likes;
            fcnn_out = self.network(im_pad)
            fcnn_depad = depad_layer(fcnn_out)[0]  # depad, and remove batch dimension;

            if margin is not None:
                fcnn_depad = fcnn_depad[margin:-margin, margin:-margin, :]
                rc_warp = rc_warp[margin:-margin, margin:-margin, :]
                im = im[:, margin:-margin, margin:-margin, :]

            # convert im to what you need for registration, since CNN has already been used;
            if self.use_postpended_channels_for_stitching:
                if stitch_rgb:
                    im = im[..., :self.num_channels_rgb]
                else:
                    im = im[..., -self.num_channels_rgb:]  # only need last few channels for stitching;
            else:
                if self.num_channels != self.num_channels_rgb:  # i.e., augmented im_stack with more channels for CNN
                    im = im[..., :self.num_channels_rgb]  # we've used the CNN; don't need the extra channels;

            ego_height = tf.reduce_mean(fcnn_depad, [-1]) * self.unet_scale  # remove feature dimension;

            # flatten out spatial dims (batch dim is 1):
            ego_height = tf.reshape(ego_height, [-1])
            rc_warp = tf.reshape(rc_warp, [-1, 2])
            im = tf.reshape(im, [-1, self.num_channels_rgb])

            H = cam_to_vanish[None]  # camera heights;
            M_j = self.magnification_j
            f_eff = self.effective_focal_length_mm

            # warp the rc_warp further using height map;
            r = rc_warp - vanish_point[None, :]  # lateral distance to vanishing point;
            delta_radial = ego_height / f_eff / (1 + 1 / M_j * H / self.H_j)  # radial deform field based on height map;
            rc = r * (1 - delta_radial[:, None]) + vanish_point[None, :]  # add back vanishing point;
            # shape of rc: _, 2;

            # stacking:
            self.im = tf.concat([im, ego_height[:, None]], axis=1)  # add height as 4th channel;

            # backprojection coordinate generation, as usual:
            # neighboring pixels:
            rc_floor = tf.floor(rc)
            rc_ceil = rc_floor + 1

            # distance to neighboring pixels:
            frc = rc - rc_floor
            crc = rc_ceil - rc

            # cast
            rc_floor = tf.cast(rc_floor, tf.int32)
            rc_ceil = tf.cast(rc_ceil, tf.int32)

            self.rc_ff = restrict(rc_floor)
            self.rc_cc = restrict(rc_ceil)
            self.rc_cf = restrict(tf.stack([rc_ceil[:, 0], rc_floor[:, 1]], 1))
            self.rc_fc = restrict(tf.stack([rc_floor[:, 0], rc_ceil[:, 1]], 1))

            self.frc = tf.exp(-frc ** 2 / 2. / self.sig_proj ** 2)
            self.crc = tf.exp(-crc ** 2 / 2. / self.sig_proj ** 2)

            # augmented coordinates:
            rc_4 = tf.concat([self.rc_ff, self.rc_cc, self.rc_cf, self.rc_fc], 0)

            # interpolated:
            im_4 = tf.concat([self.im * self.frc[:, 0, None] * self.frc[:, 1, None],
                              self.im * self.crc[:, 0, None] * self.crc[:, 1, None],
                              self.im * self.crc[:, 0, None] * self.frc[:, 1, None],
                              self.im * self.frc[:, 0, None] * self.crc[:, 1, None]], 0)
            w_4 = tf.concat([self.frc[:, 0] * self.frc[:, 1],
                             self.crc[:, 0] * self.crc[:, 1],
                             self.crc[:, 0] * self.frc[:, 1],
                             self.frc[:, 0] * self.crc[:, 1]], 0)
            # backproject:
            normalize = tf.scatter_nd(rc_4, w_4, self.recon_shape_base)
            recon = tf.scatter_nd(rc_4, im_4, [self.recon_shape_base[0], self.recon_shape_base[1],
                                                    self.num_channels_rgb + 1])
            recon_cumulative = recon_cumulative + recon
            normalize_cumulative = normalize_cumulative + normalize

        recon = tf.math.divide_no_nan(recon_cumulative, normalize_cumulative[:, :, None])

        return recon, normalize_cumulative

    def checkpoint_all_variables(self, path='./tf_ckpts', skip_saving=False, max_to_keep=2):
        # override mesoSfM's method, since we only need to keep track of the CNN variables (and optimizer);
        if self.ckpt is None:
            self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, network=self.network)
            self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=max_to_keep)
            # only keep two, restore the oldest;
        if not skip_saving:
            self.manager.save()

    def restore_all_variables(self, ckpt_no=0):
        self.ckpt.restore(self.manager.checkpoints[ckpt_no])


class fcnn(tf.keras.Model):
    # fully convolutional encoder-decoder network
    def __init__(self, filters_list, skip_list, output_nonlinearity, num_inputs):
        # filters_list and skip_list are lists of number of filters in the upsample/downsample layers,
        # and the number of filters in the skip connections;
        # output_nonlinearity can be 'leaky_relu' or 'linear';
        # num_inputs: the individual inputs are currently stacked along the channels dimension -- this tells how many
        # to look for;
        super(fcnn, self).__init__()
        assert len(filters_list) == len(skip_list)
        self.filters_list = filters_list
        self.skip_list = skip_list
        self.output_nonlinearity = output_nonlinearity
        self.num_inputs = num_inputs
        (self.downsample_list, self.downsample_skip_block_list,
         self.upsample_list, self.upsample_concat_list) = self._build()

    def _build(self, encoder_only=False):
        # define all the layers of the encoder-decoder network;
        # encoder_only: if True, only return the list of downsample layers;

        downsample_list = list()  # stores list of downsample blocks;
        downsample_skip_block_list = list()  # stores list of skip convolutional blocks;
        upsample_list = list()  # stores list of upsample blocks;
        upsample_concat_list = list()  # stores list of concatenation layers;

        # downsampling half:
        for num_filters, num_skip_filters in zip(self.filters_list, self.skip_list):
            downsample_list.append(self._downsample_block(num_filters))  # add to list of layers
            downsample_skip_block_list.append(self._skip_block(num_skip_filters))

        if encoder_only:
            return downsample_list, downsample_skip_block_list
        else:
            # upsampling half:
            for i, (num_filters, num_skip_filters) in enumerate(zip(self.filters_list[::-1], self.skip_list[::-1])):
                if num_skip_filters != 0:
                    upsample_concat_list.append(tf.keras.layers.Concatenate())
                else:
                    upsample_concat_list.append(None)  # as a placeholder
                if i == len(self.filters_list) - 1:
                    # last block, use the specified output nonlinearity:
                    upsample_list.append(self._upsample_block(num_filters,
                                                                   nonlinearity=self.output_nonlinearity))
                else:
                    upsample_list.append(self._upsample_block(num_filters))

            return downsample_list, downsample_skip_block_list, upsample_list, upsample_concat_list

    def _downsample_block(self, numfilters):
        return [tf.keras.layers.Conv2D(filters=numfilters, kernel_size=3,
                                       strides=(1, 1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                # repeat, but no downsample this time
                tf.keras.layers.Conv2D(filters=numfilters, kernel_size=1,
                                       strides=(1, 1), padding='valid'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.MaxPool2D()]

    def _upsample_block(self, numfilters, nonlinearity='leaky_relu'):
        layers_list = [tf.keras.layers.UpSampling2D(interpolation='nearest'),
                       tf.keras.layers.Conv2D(filters=numfilters, kernel_size=3,
                                              strides=(1, 1), padding='same'),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.LeakyReLU(),
                       tf.keras.layers.Conv2D(filters=numfilters, kernel_size=1,
                                              strides=(1, 1), padding='valid'),
                       tf.keras.layers.BatchNormalization()]
        if nonlinearity == 'leaky_relu':
            layers_list.append(tf.keras.layers.LeakyReLU())
        elif nonlinearity == 'linear':
            pass
        else:
            raise Exception('invalid nonlinearity')
        return layers_list

    def _skip_block(self, numfilters=4, kernel_size=1):
        if numfilters == 0:  # no skip connections
            return None
        elif numfilters == -1:  # add skip, but don't process it;
            return -1
        else:
            return [tf.keras.layers.Conv2D(filters=numfilters, kernel_size=kernel_size,
                                           strides=(1, 1), padding='valid'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU()]

    def call(self, x):
        skip_layers = list()  # store skip layer outputs to be concatenated in the upsample block;
        for down_block, skip_block in zip(self.downsample_list, self.downsample_skip_block_list):
            for down_layer in down_block:  # traverse all layers in block;
                x = down_layer(x)
            if skip_block is not None:  # if there's a skip block, traverse all layers in it;
                if skip_block != -1:
                    x_ = x
                    for skip_layer in skip_block:
                        x_ = skip_layer(x_)
                    skip_layers.append(x_)
                else:  # if -1, signifying to append skip connections without modification;
                    skip_layers.append(x)
            else:
                skip_layers.append(None)

        for up_block, skip, concat in zip(self.upsample_list, skip_layers[::-1], self.upsample_concat_list):
            if skip is not None:
                x = concat([x, skip])
            for up_layer in up_block:
                x = up_layer(x)
        return x

    def get_compatible_size(self, dim, max_dim=10000):
        # for a given dim and number of downsample blocks, find the smallest value >= dim such that the
        # network will return the same size as the input;
        # max_dim is the largest value to be considered;
        num_downsamp = len(self.filters_list)

        for bottle_neck in range(1, max_dim):
            output = bottle_neck
            # increase dims due to transposed convs:
            for j in range(num_downsamp):
                output = output * 2

            if output >= dim:
                break

        return output


@tf.function
def tf_interp(rc1, im1, rc2, sig_proj=.42465):
    # adapted from backproject_and_predict;
    # rc1/rc2 are of shape _, 2, and im1 is _,c, where c is the number of channels;

    num_channels = im1.shape[1]
    im1 = tf.cast(im1, tf.float32)
    len1 = len(rc1)

    rc = tf.concat([rc1, rc2], axis=0)  # so that one set of code can process both sets of coordinates;
    rc_max = tf.reduce_max(rc, axis=0)  # to get the bounding box
    rc_min = tf.reduce_min(rc, axis=0)
    project_shape = tf.cast(rc_max - rc_min + 4, tf.int32)  # for scatter_nd;
    rc = rc - rc_min[None, :] + 1

    # backprojection coordinate generation, as usual:
    # neighboring pixels:
    rc_floor = tf.floor(rc)
    rc_ceil = rc_floor + 1

    # distance to neighboring pixels:
    frc = rc - rc_floor
    crc = rc_ceil - rc

    # cast
    rc_floor = tf.cast(rc_floor, tf.int32)
    rc_ceil = tf.cast(rc_ceil, tf.int32)

    rc_ff = rc_floor
    rc_cc = rc_ceil
    rc_cf = tf.stack([rc_ceil[:, 0], rc_floor[:, 1]], 1)
    rc_fc = tf.stack([rc_floor[:, 0], rc_ceil[:, 1]], 1)

    frc = tf.exp(-frc ** 2 / 2. / sig_proj ** 2)
    crc = tf.exp(-crc ** 2 / 2. / sig_proj ** 2)

    # reseparate into rc1 and rc2;
    frc1 = frc[:len1]
    crc1 = crc[:len1]
    frc2 = frc[len1:]
    crc2 = crc[len1:]

    # augmented coordinates:
    rc1_4 = tf.stack([rc_ff[:len1], rc_cc[:len1], rc_cf[:len1], rc_fc[:len1]], 0)  # shape: 4, _, 2;
    rc2_4 = tf.stack([rc_ff[len1:], rc_cc[len1:], rc_cf[len1:], rc_fc[len1:]], 0)
    rc1_4 = tf.reshape(rc1_4, [-1, 2])
    rc2_4 = tf.reshape(rc2_4, [-1, 2])

    # interpolated:
    im_4 = tf.stack([im1 * frc1[:, 0, None] * frc1[:, 1, None],
                     im1 * crc1[:, 0, None] * crc1[:, 1, None],
                     im1 * crc1[:, 0, None] * frc1[:, 1, None],  # shape: _, channels;
                     im1 * frc1[:, 0, None] * crc1[:, 1, None]], 0)
    w_4 = tf.stack([frc1[:, 0] * frc1[:, 1],
                    crc1[:, 0] * crc1[:, 1],
                    crc1[:, 0] * frc1[:, 1],
                    frc1[:, 0] * crc1[:, 1]], 0)  # shape: _;
    im_4 = tf.reshape(im_4, [-1, num_channels])
    w_4 = tf.reshape(w_4, [-1])

    # backproject:
    normalize = tf.scatter_nd(rc1_4, w_4, project_shape)
    recon = tf.scatter_nd(rc1_4, im_4, tf.concat([project_shape, num_channels * tf.ones(1, dtype=tf.int32)], axis=0))
    recon = tf.math.divide_no_nan(recon, normalize[:, :, None])
    # shape: x, y, num channels;

    # now, forward prediction (interpolation onto rc2):
    gathered = tf.gather_nd(recon, rc2_4)  # shape: 4*_, channels;
    gathered = tf.reshape(gathered, (4, -1, num_channels))
    ff, cc, cf, fc = tf.unstack(gathered, num=4, axis=0)  # shape of each: _, channels;

    forward = (ff * frc2[:, 0, None] * frc2[:, 1, None] +
               cc * crc2[:, 0, None] * crc2[:, 1, None] +
               cf * crc2[:, 0, None] * frc2[:, 1, None] +
               fc * frc2[:, 0, None] * crc2[:, 1, None])

    forward /= ((frc2[:, 0, None] * frc2[:, 1, None]) +
                (crc2[:, 0, None] * crc2[:, 1, None]) +
                (crc2[:, 0, None] * frc2[:, 1, None]) +
                (frc2[:, 0, None] * crc2[:, 1, None]))  # shape: _, channels;
    return forward

def create_multiview_stack(im_stack, rc_warp, connectivity=np.ones((3, 5))):
    # im_stack: e.g. of shape 9, 6, 3120, 4096, 3;  # color channel must be present;
    # rc_warp: e.g. of shape 9, 6, 3120, 4096, 2; coordinates of every pixel after homographic warping;
    # connectivity specifies which neighboring cameras to consider;
    # returns an augmented stack of shape, e.g., 9, 6, 3120, 4096, 3*9 (or 3*3 -- 3 times (number of neighbors + 1));

    num_row, num_col, H, W, num_channels = im_stack.shape
    assert (connectivity.shape[0] % 2 == 1) and (connectivity.shape[1] % 2 == 1)  # must be odd;

    # center must be 1:
    if connectivity.shape[0] == 1:
        center_r = 0
    else:
        center_r = connectivity.shape[0] // 2 + 1
    if connectivity.shape[1] == 1:
        center_c = 0
    else:
        center_c = connectivity.shape[1] // 2 + 1
    assert connectivity[center_r, center_c] == 1

    num_neigh = np.int32(np.sum(connectivity)) - 1  # exclude the center;

    im_stack_augmented = np.zeros((num_row, num_col, H, W, num_channels * num_neigh), dtype=np.uint8)
    for row in tqdm(range(num_row)):
        for col in range(num_col):  # outer two loops: loop over all cameras indicated by connectivity;
            row_sweep = np.arange(connectivity.shape[0]) - connectivity.shape[0] // 2 + row
            col_sweep = np.arange(connectivity.shape[1]) - connectivity.shape[1] // 2 + col

            counter = 0
            for i_col in range(len(col_sweep)):
                for i_row in range(len(row_sweep)):  # inner two loops: loop over the neighbors
                    row_neigh = row_sweep[i_row]
                    col_neigh = col_sweep[i_col]
                    consider_this_cam = (connectivity[i_row, i_col] == 1)  # whether to consider this cam;

                    if consider_this_cam:
                        if row_neigh == row and col_neigh == col:
                            # the image itself; skip, and don't increment counter, because this image is never included;
                            pass
                        elif (row_neigh < 0) or (col_neigh < 0) or (row_neigh >= num_row) or (col_neigh >= num_col):
                            # neighbor doesn't exist; skip, but still leave the channels blank (0s), as other cameras
                            # may have this neighbor;
                            counter += num_channels
                        else:
                            im_interp = tf_interp(rc_warp[row_neigh, col_neigh].reshape(-1, 2),
                                                  im_stack[row_neigh, col_neigh].reshape(-1, num_channels),
                                                  rc_warp[row, col].reshape(-1, 2)).numpy()  # resampling/interp;
                            im_stack_augmented[row, col, :, :, counter:counter + num_channels] = np.uint8(
                                im_interp.reshape(H, W, num_channels))

                            counter += num_channels
                    else: # don't increment counter, because this would be a waste, as these channels will always be 0;
                        pass

    # always make the center image the first channel:
    im_stack_augmented = np.concatenate([im_stack, im_stack_augmented], axis=4)

    return im_stack_augmented


def edge_filter_rgb_stack(stack, sigmas=(1, 3, 9), keep_rgb=False, gaussian_only=False):
    # takes in an rgb stack and returns an edged-filtered version;
    # expect stack of shape, e.g., 54, 3120, 4096, 3;
    # based on scipy.ndimage.gaussian_laplace;
    # sigmas is a list of sigma values for gaussian_laplace; stacked in channel dimension;
    # keep_rgb: if true, stack new result on top of rgb;

    # convert to rgb:
    gray = np.sum(stack.astype(np.float32) * np.array([0.2126, 0.7152, 0.0722],
                                                      dtype=np.float32)[None, None, None, :], axis=3)

    if gaussian_only:
        filt = scipy.ndimage.gaussian_filter
    else:
        filt = scipy.ndimage.gaussian_laplace

    new_stack = list()
    for sigma in sigmas:
        # filter each image in stack with laplacian:
        gray_filt = np.stack([filt(im, sigma=sigma) for im in gray])

        # convert back to uint8:
        gray_filt *= sigma ** 2  # the larger sigma, the smaller the peak value;
        gray_filt = (gray_filt - np.min(gray_filt))
        gray_filt = (gray_filt / np.max(gray_filt) * 255).astype(np.uint8)  # try to use up dynamic range;

        max_val = gray_filt.max()
        if max_val > 255:
            print('Warning: max value of new stack is ' + str(max_val))

        new_stack.append(gray_filt)
    new_stack = np.stack(new_stack, axis=3)  # should have channel dim even if length of sigams is 1;

    if keep_rgb:
        return np.concatenate([new_stack, stack], axis=3)
    else:
        return new_stack


def edge_filter_rgb_stack_normalized_hpf(stack, sigmas=(1, 3, 9), keep_rgb=False):
    # similar to edge_filter_rgb_stack_normalized, but using a different edge filter;

    # convert to rgb:
    gray = np.sum(stack.astype(np.float32) * np.array([0.2126, 0.7152, 0.0722],
                                                      dtype=np.float32)[None, None, None, :], axis=3)

    new_stack = list()
    for sigma in sigmas:
        # filter each image in stack with laplacian:
        # gray_filt = np.stack([im / scipy.ndimage.gaussian_filter(im, sigma=sigma) for im in gray])

        # this is based on a difference-of-gaussian approx to scipy.ndimage.gaussian_laplace, except here we're dividing
        # instead of taking the difference:
        gray_filt = np.stack([scipy.ndimage.gaussian_filter(im, np.sqrt(2)*sigma) /
                              scipy.ndimage.gaussian_filter(im, sigma) for im in gray])

        # try to use up dynamic range:
        gray_filt = (-gray_filt + 1) / sigma * 3000 + 128  # determined the 3000 value empirically
        gray_filt = np.clip(gray_filt, 0, 255).astype(np.uint8)
        # ^since these images are normalized absolutely, be sure to scale in the same way for all images when rescaling;

        new_stack.append(gray_filt)
    new_stack = np.stack(new_stack, axis=3)  # should have channel dim even if length of sigams is 1;

    if keep_rgb:
        return np.concatenate([new_stack, stack], axis=3)
    else:
        return new_stack


def flatten_illumination(im_stack, inds_keep, illum_flat):
    # corrects illumination unevenness based on optimized illumination parameters (illum_flat);

    im_stack_inds_keep = im_stack[inds_keep]

    c = np.arange(im_stack.shape[2], dtype=np.uint16)
    r = np.arange(im_stack.shape[1], dtype=np.uint16)
    r, c = np.meshgrid(r, c, indexing='ij')
    rc_base = np.stack([r, c]).T
    rc_base = np.tile(rc_base[None], [len(inds_keep), 1, 1, 1])

    rc = np.transpose(rc_base, (0, 2, 1, 3))
    rc = np.reshape(rc, (rc.shape[0], -1, rc.shape[-1]))  # flatten spatial dims;

    im_dims = np.array(im_stack.shape)[1:3]  # for normalization of image coordinates;
    max_dim = np.max(im_dims)  # to keep isotropic;
    camera_yx = (rc - .5 * im_dims[None, None, :]) / max_dim

    y = camera_yx[:, :, 0]
    x = camera_yx[:, :, 1]
    DC = illum_flat[:, 0:1]
    DC = DC - np.mean(DC)  # to avoid global gain;
    correction = (1 + DC + illum_flat[:, 1:2] * x + illum_flat[:, 2:3] * y
                  + illum_flat[:, 3:4] * x ** 2 + illum_flat[:, 4:5] * y ** 2 + illum_flat[:, 5:] * x * y)
    # shape: num_images, _;
    correction = correction.reshape([len(inds_keep)] + list(im_dims) + [1])  # unflatten image dims;

    im_stack_inds_keep = im_stack_inds_keep * correction

    im_stack_inds_keep = np.clip(im_stack_inds_keep, 0, 255)

    return im_stack_inds_keep


def load_stack(directory, filename, video_frame=None):
    # directory + filename is a .nc file containing a single mcam frame or a sequence;
    # if a sequence,then need to specify a video_frame; can only be one;

    if filename == '':
        nc_path = directory
    else:
        nc_path = os.path.join(directory, filename)

    if video_frame is None:
        data = xr.open_dataset(nc_path, engine='netcdf4')
    else:
        # for some reason, this is a lot faster (requires dask)
        data = xr.open_dataset(nc_path, engine='netcdf4', chunks={'frame_number': 1})

    if 'mcam_data' in data:
        mcam_data = data.mcam_data
    elif 'images' in data:
        mcam_data = data.images
    else:
        raise Exception('invalid dataset')

    if len(mcam_data.shape) == 5:
        assert video_frame is not None
        im_stack = mcam_data.data[video_frame].compute(scheduler='single-threaded')
    elif len(mcam_data.shape) == 4:
        im_stack = mcam_data.data
    else:
        raise Exception('invalid mcam shape: ' + str(data.mcam_data.shape))

    im_stack = im_stack.reshape(-1, im_stack.shape[2], im_stack.shape[3])
    im_stack = np.stack([cv2.cvtColor(im, cv2.COLOR_BAYER_GB2BGR) for im in im_stack])  # debayer

    return im_stack
