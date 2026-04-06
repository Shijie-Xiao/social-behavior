import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc
import sys
from torch.utils.data import Dataset, DataLoader



class Visualiser:
    def __init__(self, frame_width, frame_height, file_dir, colors = ('lawngreen', 'skyblue', 'tomato'), 
                 tcolors = ('green', 'blue', 'red'), animation = True, fps = 5):
        '''
        Class to visualize the mouse trajectories with animation enabled. 
        
        '''

        self.FRAME_WIDTH_TOP = frame_width
        self.FRAME_HEIGHT_TOP = frame_height
        self.M1_COLOR, self.M2_COLOR, self.M3_COLOR = colors
        self.M1_TCOLOR, self.M2_TCOLOR, self.M3_TCOLOR = tcolors

        self.PLOT_MOUSE_START_END = [(0, 1), (1, 3), (3, 2), (2, 0),        # head
                                     (3, 6), (6, 9),                        # midline
                                     (9, 10), (10, 11),                     # tail
                                     (4, 5), (5, 8), (8, 9), (9, 7), (7, 4) # legs
                                    ]
        self.file_dir = file_dir
        self.fps = fps

        if animation: 
            rc('animation', html='jshtml')
        
    def set_figax(self):
        fig = plt.figure(figsize=(8, 8))
        img = np.ones((self.FRAME_HEIGHT_TOP, self.FRAME_WIDTH_TOP, 3))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        return fig, ax
    
    def plot_mouse(self, ax, pose, color):
        # Draw each keypoint
        for j in range(10):
            ax.plot(pose[j, 0], pose[j, 1], 'o', color = color, markersize=3)

        # Draw a line for each point pair to form the shape of the mouse

        for pair in self.PLOT_MOUSE_START_END:
            line_to_plot = pose[pair, :]
            ax.plot(line_to_plot[:, 0], line_to_plot[
                    :, 1], color=color, linewidth=1)
    
    def track(self, ax, pose, color, kp = 6):
        frame = pose[:,kp, :]
        ax.plot(frame[:, 0], frame[:, 1], color = color, linewidth = 2)


    def animate_pose_sequence(self, seq, start_frame = 0, stop_frame = 100, skip = 0):
        '''
        Returns the animation of the keypoint sequence between start frame
        and stop frame. Optionally can display annotations.
        '''

        image_list = []

        counter = 0
        if skip:
            anim_range = range(start_frame, stop_frame, skip)
        else:
            anim_range = range(start_frame, stop_frame)

        for j in anim_range:
            if counter % 20 == 0:
                print("Processing frame ", j)
            fig, ax = self.set_figax()
            self.plot_mouse(ax, seq[j, 0, :, :], color = self.M1_COLOR)
            self.plot_mouse(ax, seq[j, 1, :, :], color = self.M2_COLOR)
            self.plot_mouse(ax, seq[j, 2, :, :], color = self.M3_COLOR)

            self.track(ax, seq[:j, 0, :, :], color = self.M1_TCOLOR)
            self.track(ax, seq[:j, 1, :, :], color = self.M2_TCOLOR)
            self.track(ax, seq[:j, 2, :, :], color = self.M3_TCOLOR)

            ax.axis('off')
            fig.tight_layout(pad=0)
            ax.margins(0)

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(),
                                            dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,))

            image_list.append(image_from_plot)

            plt.close()
            counter = counter + 1

        # Plot animation.
        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        im = plt.imshow(image_list[0])

        def animate(k):
            im.set_array(image_list[k])
            return im,
        ani = animation.FuncAnimation(fig, animate, frames=len(image_list), blit=True)
        return ani
    
    def fill_holes(self, data):
        '''
        Function to handle empty frames / discontinuities while processing sequences
        
        '''
        clean_data = data.copy()
        for m in range(3):
            holes = np.where(clean_data[0,m,:,0] == -1)
            if not holes:
                continue
            for h in holes[0]:
                sub = np.where(clean_data[:,m,h,0] != -1)
                if(sub and sub[0].size > 0):
                    clean_data[0,m,h,:] = clean_data[sub[0][0],m,h,:]
                else:
                    return np.empty((0))

        for fr in range(1,np.shape(clean_data)[0]):
            for m in range(3):
                holes = np.where(clean_data[fr,m,:,0] == -1)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
        return clean_data
    
    def animate(self, data, fname, skip_frames = 0): 
        # Ensure directory exists
        os.makedirs(self.file_dir, exist_ok=True)
        fpath = os.path.join(self.file_dir, fname)
        start_frame, stop_frame = 0, len(data) - 1
        clean_data = self.fill_holes(data)
        ani = self.animate_pose_sequence(
                                         clean_data,
                                         start_frame = start_frame,
                                         stop_frame = stop_frame,
                                         skip = skip_frames)
        
        try:
            ani.save(fpath + '.mp4',writer = 'ffmpeg',fps = self.fps)
            print("Data saved to {}".format(fpath + '.mp4'))
        except Exception as e:
            print("Warning: Failed to save video with ffmpeg: {}".format(e))
            print("Attempting to save as GIF instead...")
            try:
                ani.save(fpath + '.gif',writer = 'pillow',fps = self.fps)
                print("Data saved to {}".format(fpath + '.gif'))
            except Exception as e2:
                print("Warning: Failed to save as GIF: {}".format(e2))
                print("Saving first frame as image instead...")
                try:
                    # Save first frame as image
                    fig, ax = self.set_figax()
                    self.plot_mouse(ax, clean_data[0, 0, :, :], color = self.M1_COLOR)
                    self.plot_mouse(ax, clean_data[0, 1, :, :], color = self.M2_COLOR)
                    self.plot_mouse(ax, clean_data[0, 2, :, :], color = self.M3_COLOR)
                    plt.savefig(fpath + '.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print("First frame saved to {}".format(fpath + '.png'))
                except Exception as e3:
                    print("Error: Could not save visualization: {}".format(e3))


class MouseMotionData(Visualiser): 
    def __init__(self, fpath, batch_size = 50, seq_length = 5, forcePreProcess = False, infer = False,\
                 window = 5, val_fraction = 0.45, frame_height = 850, frame_width = 850, rh = 2, rw = 2, motion_threshold = 500, frame_subsample = 4):
        '''
        DataLoader object to load the coordinates of the mouse triplet dataset and containerize them as torch.Tensor. For the sake of simplicity and validation, 
        only mouse behaviour when lights are switched off is studied.

        params:
        frame_subsample : int, take every Nth frame (default: 4). When > 1: (1) reduces
            keypoint jitter from high-fps pose estimation; (2) increases per-frame
            displacement so temporal edges encode meaningful velocity; (3) aligns
            effective fps with ETH/UCY (~2.5 fps). Use 1 only if source is low-fps.
        '''
        # Set directory first
        self.DIR = os.path.dirname(fpath)
        
        # Set video directory relative to data directory
        video_dir = os.path.join(self.DIR, "videos")
        super().__init__(rh, rw, video_dir)

        # Load data - handle both dict and array formats
        loaded_data = np.load(fpath, allow_pickle = True)
        if isinstance(loaded_data, np.ndarray):
            # If it's a numpy array, convert to dict format
            print("Warning: Data is a numpy array. Converting to dict format...")
            # Assume array contains sequences directly: shape should be (num_sequences, num_frames, num_mice, num_keypoints, 2)
            # Convert to expected dict format
            if loaded_data.ndim >= 4:
                # Create sequences dict
                sequences_dict = {}
                for i in range(len(loaded_data)):
                    sequences_dict[i] = {
                        'keypoints': loaded_data[i],
                        'annotations': [np.ones(loaded_data[i].shape[0]), np.zeros(loaded_data[i].shape[0])]
                    }
                self.rawData = {'sequences': sequences_dict}
            else:
                # Try .item() for object arrays
                self.rawData = loaded_data.item() if hasattr(loaded_data, 'item') else {'sequences': {0: {'keypoints': loaded_data, 'annotations': [np.ones(len(loaded_data)), np.zeros(len(loaded_data))]}}}
        elif isinstance(loaded_data, dict):
            self.rawData = loaded_data
        else:
            # Try .item() for numpy scalar/object types
            self.rawData = loaded_data.item() if hasattr(loaded_data, 'item') else loaded_data
        self.window = window

        self.infer = infer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.val_fraction = val_fraction
        self.numMice = 3
        self.FRAME_HEIGHT, self.FRAME_WIDTH = frame_height, frame_width
        self.motion_threshold = motion_threshold
        self.frame_subsample = frame_subsample  # Subsample frames: take every Nth frame

        trajectories_file = os.path.join(self.DIR, "trajectories_fs{}.cpkl".format(self.frame_subsample))

        if not (os.path.exists(trajectories_file)) or forcePreProcess: 
            print("Creating pre-processed data from raw data")
            self.frame_preprocess(trajectories_file)
        
        self.load_preprocessed(trajectories_file)

        self.reset_batch_pointer(valid = True)
        self.reset_batch_pointer(valid = False)

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0            
    
    def frame_preprocess(self, data_file):
        '''
        Pre-process MaBE dataset into train/valid splits.
        Uses sequence-level split (not per-frame) so validation has enough data.
        '''
        all_frame_data = []
        valid_frame_data = []
        frameList_data = []

        self.dataset, lframes, dframes = self.getActiveMiceData(self.rawData)
        print('Total lit sequences where the motion is observed: {}'.format(lframes))
        print('Total non-lit sequences where the motion is observed: {}'.format(dframes))

        sequenceIDs = list(self.dataset.keys())
        n_seq = len(sequenceIDs)

        # Sequence-level split: assign whole sequences to train or valid
        # Ensures validation has enough frames (per-frame split often yields 0 valid batches for short chunks)
        np.random.seed(42)
        perm = np.random.permutation(n_seq)
        n_valid = max(1, int(n_seq * self.val_fraction))
        valid_ids = set(perm[:n_valid])
        train_ids = set(perm[n_valid:])

        for seq_idx, sId in enumerate(sequenceIDs):
            data = self.dataset[sId]
            frameList = np.arange(data.shape[0])
            numFrames = len(frameList)

            frameList_data.append(frameList)
            all_frame_data.append([])
            valid_frame_data.append([])

            is_valid_seq = seq_idx in valid_ids
            for ind, frame in enumerate(frameList):
                miceWithPos = []
                selected_kps = [0, 1, 2, 7, 10]

                for mice in range(self.numMice):
                    for kp_idx, kp_origin_idx in enumerate(selected_kps):
                        if data.shape[2] > kp_origin_idx:
                            current_x = float(data[frame, mice, kp_origin_idx, 0])
                            current_y = float(data[frame, mice, kp_origin_idx, 1])
                        else:
                            current_x = float(data[frame, mice, 0, 0])
                            current_y = float(data[frame, mice, 0, 1])
                        virtual_node_id = mice * len(selected_kps) + kp_idx
                        miceWithPos.append([virtual_node_id, current_x, current_y])

                arr = np.array(miceWithPos)
                if self.infer:
                    all_frame_data[seq_idx].append(arr)
                elif is_valid_seq:
                    valid_frame_data[seq_idx].append(arr)
                else:
                    all_frame_data[seq_idx].append(arr)

        n_valid_frames = sum(len(v) for v in valid_frame_data)
        n_train_frames = sum(len(a) for a in all_frame_data)
        print('Train: {} seqs, {} frames; Valid: {} seqs, {} frames'.format(
            len(train_ids), n_train_frames, len(valid_ids), n_valid_frames))

        f = open(data_file, "wb")
        pickle.dump((all_frame_data, frameList_data, valid_frame_data), f, protocol=2)
        f.close() 

    def getDataSlice(self, data, begin, indices): 
        start = end = indices[begin]  
        seqLen = 0
        i = begin + 1

        while i < len(indices) and indices[i] == end + 1:
            seqLen += 1
            end = indices[i]
            i += 1    

        start_idx = max(0, start - self.window) 
        end_idx = min(len(data), end + self.window + 1) 
        # print(start_idx,end_idx)

        slicedData = data[start_idx:end_idx]

        return slicedData, end, i


    def fill_holes(self, data):
        clean_data = data.copy()
        for m in range(3):
            holes = np.where(clean_data[0,m,:,0] == -1)
            if not holes:
                continue
            for h in holes[0]:
                sub = np.where(clean_data[:,m,h,0] != -1)
                if(sub and sub[0].size > 0):
                    clean_data[0,m,h,:] = clean_data[sub[0][0],m,h,:]
                else:
                    return np.empty((0))

        for fr in range(1,np.shape(clean_data)[0]):
            for m in range(3):
                holes = np.where(clean_data[fr,m,:,0] == -1)
                if not holes:
                    continue
                for h in holes[0]:
                    clean_data[fr,m,h,:] = clean_data[fr-1,m,h,:]
        return clean_data


    def normalize(self, data):
        # Map pixel coords to [0, 10] for Gaussian Likelihood gradient stability (ETH/UCY scale)
        data = (data / self.FRAME_HEIGHT) * 10.0
        return data
    
    def getActiveMiceData(self, dataset): 
        '''
        Helper function to get the active samples (where mice moves/interacts with other mice or the environment). Mice movement is maximized 
        when the lights are switched off
        '''
        sequences = dataset['sequences']
        data = {}

        counter = 0
        nlightframes, ndarkframes = 0, 0
        for sequence in sequences.keys(): 
            seq = np.array(sequences[sequence].get('keypoints'))  
            ann_chase = np.array(sequences[sequence].get('annotations')[0])
            
            # Frame subsampling: take every Nth frame. Purpose: (1) reduce noise from
            # high-fps keypoint jitter; (2) increase per-frame displacement so temporal
            # edges encode meaningful velocity; (3) align with ETH/UCY ~2.5fps scale.
            # Use frame_subsample=1 only if source video is already low-fps.
            if self.frame_subsample > 1:
                # Subsample the sequence data
                seq = seq[::self.frame_subsample]
                # Subsample the annotations accordingly
                ann_chase = ann_chase[::self.frame_subsample]
                print("Sequence {}: Subsampled from {} frames to {} frames (subsample rate: {})".format(
                    sequence, len(sequences[sequence].get('keypoints')), len(seq), self.frame_subsample))
            
            chaseInd = self.getChaseInd(seq, ann_chase, motion_threshold=self.motion_threshold)
            # print(chaseInd)
            lightInd = np.where(sequences[sequence].get('annotations')[1] == 0)[-1]
            if not len(chaseInd): 
                continue 
            
            if len(lightInd): nlightframes += 1
            else: ndarkframes += 1

            start, end = chaseInd[0], -1
            # Use subsampled seq for slicing (do not overwrite with original keypoints)
            start = 0
            
            while end != chaseInd[-1]: 
                slicedData, end, start = self.getDataSlice(seq, start, chaseInd)
                ndata = self.normalize(slicedData)
                filled_data = self.fill_holes(ndata)

                if filled_data.shape[0] == 0:
                    print("Skipping empty filled data")
                    continue  

                data[counter] = filled_data

                counter += 1  

        
        print("Processed {} valid sequences.".format(len(data)))
        return data, nlightframes, ndarkframes

    
    def getChaseInd(self,seq, ann_chase, motion_threshold=20):
        T = seq.shape[0]  # Number of frames
        M = seq.shape[1]  # Number of mice
        K = seq.shape[2]  # Number of keypoints per mouse
        window_size = self.seq_length // 2

        # **Step 1: Select frames where annotations indicate chasing**
        chase_label_ind = np.where(ann_chase == 1)[0]

        # **Step 2: Apply window expansion around chase frames only (no motion filtering)**
        final_frames = set()
        for frame in chase_label_ind:
            start = max(0, frame - window_size)
            end = min(T, frame + window_size)
            final_frames.update(range(start, end))
        
        # **Step 3: Return sorted unique frame indices**
        return np.array(sorted(final_frames))


    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.train_data = raw_data[0]
        self.frameList  = raw_data[1]
        self.valid_data = raw_data[2]
        
        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.train_data)):
            # get the frame data for the current dataset
            all_frame_data = self.train_data[dataset]
            valid_frame_data = self.valid_data[dataset]
            # print('Training data from dataset {} : {}'.format(dataset, len(all_frame_data)))
            # print('Validation data from dataset {} : {}'.format(dataset, len(valid_frame_data)))

            counter += int(len(all_frame_data) / (self.seq_length))
            valid_counter += int(len(valid_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)

        print('Total number of training batches: {}'.format(self.num_batches * 2))
        print('Total number of validation batches: {}'.format(self.valid_num_batches))
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2

    
    def next_batch(self, randomUpdate=True):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Frame data
        frame_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            
            frame_data = self.train_data[self.dataset_pointer]

            frame_ids = self.frameList[self.dataset_pointer]

            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]
                seq_frame_ids = frame_ids[idx:idx+self.seq_length]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                frame_batch.append(seq_frame_ids)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.frame_pointer += np.random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, frame_batch, d

    def next_valid_batch(self, randomUpdate=True):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += np.random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d
    
    
    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.train_data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0  

    
    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0

    def visualize(self, fname):
        '''
        Animate a random sequence to verify whether the constructed dataset has no discontinuities. 
        '''
        idx = np.random.randint(0, len(self.train_data))
        data = np.array(self.train_data[idx])
        self.animate(data[:,:,:,1:], fname)


def main():
    '''
    Main function to test data loading, preprocessing, and visualization
    '''
    import argparse
    
    parser = argparse.ArgumentParser(description='Test mouse data loading and visualization')
    parser.add_argument('--data_path', type=str, 
                       default='../data/mice/user_train_r1.npy',
                       help='Path to the mouse data file')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for testing')
    parser.add_argument('--seq_length', type=int, default=20,
                       help='Sequence length')
    parser.add_argument('--force_preprocess', action='store_true',
                       help='Force preprocessing even if trajectories.cpkl exists')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization animation')
    parser.add_argument('--test_batches', type=int, default=3,
                       help='Number of batches to test loading')
    
    args = parser.parse_args()
    
    # Get absolute path
    if not os.path.isabs(args.data_path):
        # If relative path, make it relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, args.data_path)
    else:
        data_path = args.data_path
    
    print("=" * 60)
    print("Mouse Data Loading and Testing Script")
    print("=" * 60)
    print("Data path: {}".format(data_path))
    print("Batch size: {}".format(args.batch_size))
    print("Sequence length: {}".format(args.seq_length))
    print("Force preprocessing: {}".format(args.force_preprocess))
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print("Error: Data file not found at {}".format(data_path))
        return
    
    # Initialize data loader
    print("\n[1/4] Initializing MouseMotionData...")
    try:
        dataloader = MouseMotionData(
            fpath=data_path,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            forcePreProcess=args.force_preprocess,
            infer=False
        )
        print("Data loader initialized successfully")
    except Exception as e:
        print(f"Error initializing data loader: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check trajectories file
    trajectories_file = os.path.join(os.path.dirname(data_path), "trajectories_fs{}.cpkl".format(dataloader.frame_subsample))
    if os.path.exists(trajectories_file):
        print(f"\nTrajectories file created at: {trajectories_file}")
        file_size = os.path.getsize(trajectories_file) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")
    else:
        print(f"\nWarning: Trajectories file not found at: {trajectories_file}")
    
    # Print data statistics
    print("\n[2/4] Data Statistics:")
    print("  Total training sequences: {}".format(len(dataloader.train_data)))
    print("  Total validation sequences: {}".format(len(dataloader.valid_data)))
    print("  Number of training batches: {}".format(dataloader.num_batches))
    print("  Number of validation batches: {}".format(dataloader.valid_num_batches))
    
    # Test batch loading
    print("\n[3/4] Testing batch loading ({} batches)...".format(args.test_batches))
    try:
        dataloader.reset_batch_pointer(valid=False)
        for i in range(args.test_batches):
            x_batch, y_batch, frame_batch, d = dataloader.next_batch(randomUpdate=False)
            print("  Batch {}:".format(i+1))
            print("    - Number of sequences: {}".format(len(x_batch)))
            if len(x_batch) > 0:
                print("    - Sequence shape: {}".format(np.array(x_batch[0]).shape))
                print("    - Frame IDs: {}...".format(frame_batch[0][:5]) if len(frame_batch[0]) > 5 else "    - Frame IDs: {}".format(frame_batch[0]))
        print("Batch loading test successful")
    except Exception as e:
        print(f"Error during batch loading: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test validation batch loading
    print("\n[4/4] Testing validation batch loading...")
    try:
        dataloader.reset_batch_pointer(valid=True)
        if dataloader.valid_num_batches > 0:
            x_batch, y_batch, d = dataloader.next_valid_batch(randomUpdate=False)
            print(f"  Validation batch:")
            print(f"    - Number of sequences: {len(x_batch)}")
            if len(x_batch) > 0:
                print(f"    - Sequence shape: {np.array(x_batch[0]).shape}")
            print("Validation batch loading test successful")
        else:
            print("  No validation batches available")
    except Exception as e:
        print(f"Error during validation batch loading: {e}")
        import traceback
        traceback.print_exc()
    
    # Visualization
    if args.visualize:
        print("\n[5/5] Creating visualization...")
        try:
            # Create output directory for videos
            video_dir = os.path.join(os.path.dirname(data_path), "videos")
            os.makedirs(video_dir, exist_ok=True)
            
            # Visualize a random sequence
            dataloader.visualize("test_sequence")
            print(f"Visualization saved to: {video_dir}/test_sequence.mp4")
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Testing completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()