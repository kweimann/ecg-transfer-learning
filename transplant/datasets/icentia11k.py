import os

import numpy as np

from transplant.utils import load_pkl

ds_segment_size = 2 ** 20 + 1   # 1,048,577
ds_frame_size = 2 ** 11 + 1   # 2,049
ds_patient_ids = np.arange(11000)
ds_sampling_rate = 250
ds_mean = 0.0018  # mean over entire dataset
ds_std = 1.3711   # std over entire dataset

ds_beat_names = {
    0: 'undefined',     # Undefined
    1: 'normal',        # Normal
    2: 'pac',           # ESSV (PAC)
    3: 'aberrated',     # Aberrated
    4: 'pvc'            # ESV (PVC)
}

ds_rhythm_names = {
    0: 'undefined',     # Null/Undefined
    1: 'end',           # End (essentially noise)
    2: 'noise',         # Noise
    3: 'normal',        # NSR (normal sinusal rhythm)
    4: 'afib',          # AFib
    5: 'aflut'          # AFlutter
}

_HI_PRIO_RHYTHMS = [0, 4, 5]  # undefined / afib / aflut
_LO_PRIO_RHYTHMS = [1, 2, 3]  # end / noise / normal

_HI_PRIO_BEATS = [2, 3, 4]  # pac / aberrated / pvc
_LO_PRIO_BEATS = [0, 1]  # undefined / normal

_HR_TACHYCARDIA = 0
_HR_BRADYCARDIA = 1
_HR_NORMAL = 2
_HR_NOISE = 3

ds_hr_names = {
    _HR_TACHYCARDIA: 'tachy',
    _HR_BRADYCARDIA: 'brady',
    _HR_NORMAL: 'normal',
    _HR_NOISE: 'noise'
}


def rhythm_data_generator(patient_generator, frame_size=2048, samples_per_patient=1):
    """
    Generate a stream of short signals and their corresponding rhythm label. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the rhythm durations within this frame.

    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.

    @return: Generator of: input data of shape (frame_size, 1), output data as the corresponding rhythm label.
    """
    for _, (signal, labels) in patient_generator:
        num_segments, segment_size = signal.shape
        patient_rhythm_labels = labels['rtype']  # note: variables in a .npz file are only loaded when accessed
        for _ in range(samples_per_patient):
            # randomly choose a frame that lies within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            x = signal[segment_index, frame_start:frame_end]
            x = np.expand_dims(x, axis=1)  # add channel dimension
            # calculate the durations of each rhythm in the frame and determine the final label
            rhythm_ends, rhythm_labels = patient_rhythm_labels[segment_index]
            frame_rhythm_durations, frame_rhythm_labels = get_rhythm_durations(
                rhythm_ends, rhythm_labels, frame_start, frame_end)
            y = get_rhythm_label(frame_rhythm_durations, frame_rhythm_labels)
            yield x, y


def beat_data_generator(patient_generator, frame_size=2048, samples_per_patient=1):
    """
    Generate a stream of short signals and their corresponding beat label. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the beats within this frame.

    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.

    @return: Generator of: input data of shape (frame_size, 1), output data as the corresponding beat label.
    """
    for _, (signal, labels) in patient_generator:
        num_segments, segment_size = signal.shape
        patient_beat_labels = labels['btype']  # note: variables in a .npz file are only loaded when accessed
        for _ in range(samples_per_patient):
            # randomly choose a frame that lies within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            x = signal[segment_index, frame_start:frame_end]
            x = np.expand_dims(x, axis=1)  # add channel dimension
            # calculate the count of each beat type in the frame and determine the final label
            beat_ends, beat_labels = patient_beat_labels[segment_index]
            _, frame_beat_labels = get_complete_beats(
                beat_ends, beat_labels, frame_start, frame_end)
            y = get_beat_label(frame_beat_labels)
            yield x, y


def heart_rate_data_generator(patient_generator, frame_size=2048, label_frame_size=None,
                              samples_per_patient=1):
    """
    Generate a stream of short signals and their corresponding heart rate label. These short signals are uniformly
    sampled from the segments in patient data by placing a frame in a random location within one of the segments.
    The corresponding label is then determined based on the beats within this frame.

    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
    @param frame_size: Size of the frame that contains a short input signal.
    @param label_frame_size: Size of the frame centered on the input signal frame, that contains a short signal used
            for determining the label. By default equal to the size of the input signal frame.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.

    @return: Generator of: input data of shape (frame_size, 1),
    output data as the corresponding heart rate label.
    """
    if label_frame_size is None:
        label_frame_size = frame_size
    max_frame_size = max(frame_size, label_frame_size)
    for _, (signal, labels) in patient_generator:
        num_segments, segment_size = signal.shape
        patient_beat_labels = labels['btype']  # note: variables in a .npz file are only loaded when accessed
        for _ in range(samples_per_patient):
            # randomly choose a point within a segment and span a frame centered on this point
            #  the frame must lie within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_center = np.random.randint(segment_size - max_frame_size) + max_frame_size // 2
            signal_frame_start = frame_center - frame_size // 2
            signal_frame_end = frame_center + frame_size // 2
            x = signal[segment_index, signal_frame_start:signal_frame_end]
            x = np.expand_dims(x, axis=1)  # add channel dimension
            # get heart rate label based on the rr intervals in an area around the frame center
            #  determined by the label frame size
            label_frame_start = frame_center - label_frame_size // 2
            label_frame_end = frame_center + label_frame_size // 2
            beat_ends, _ = patient_beat_labels[segment_index]
            frame_beat_ends = get_complete_beats(beat_ends, start=label_frame_start, end=label_frame_end)
            y = get_heart_rate_label(frame_beat_ends, ds_sampling_rate)
            yield x, y


def signal_generator(patient_generator, frame_size=2048, samples_per_patient=1):
    """
    Generate a stream of short signals. These short signals are uniformly sampled
    from the segments in patient data by placing a frame in a random location within one of the segments.

    @param patient_generator: Generator that yields a tuple of patient id and patient data at each iteration.
            Patient data may contain only signals, since labels are not used.
    @param frame_size: Size of the frame that contains a short signal.
    @param samples_per_patient: Number of samples from one patient before new patient is pulled from the generator.
            This is done in order to decrease the number of i/o operations.

    @return: Generator of: input data of shape (frame_size, 1)
    """
    for _, (signal, _) in patient_generator:
        num_segments, segment_size = signal.shape
        for _ in range(samples_per_patient):
            # randomly choose a frame that lies within the segment i.e. no zero-padding is necessary
            segment_index = np.random.randint(num_segments)
            frame_start = np.random.randint(segment_size - frame_size)
            frame_end = frame_start + frame_size
            x = signal[segment_index, frame_start:frame_end]
            x = np.expand_dims(x, axis=1)  # add channel dimension
            yield x


def cpc_data_generator(buffered_patient_generator, context_size, ns, frame_size=2048, context_overlap=0,
                       positive_offset=0, ns_same_segment_prob=None, samples_per_patient=1):
    """
    Generate a stream of input, output data for the Contrastive Predictive Coding
    (Oord et al. (2018) https://arxiv.org/abs/1807.03748) representation learning approach.
    The idea is to predict the positive sample from the future among distractors (negative samples)
    based on some context. Note, that while we attempt to capture the idea of the proposed approach for 1D data,
    our implementation deviates from the implementation described in the paper.

    @param buffered_patient_generator: Generator that yields a buffer filled with patient data at each iteration.
            Patient data may contain only signals, since labels are not used.
    @param context_size: Number of frames that make up the context.
    @param ns: Number of negative samples.
    @param frame_size: Size of the frame that contains a short signal.
    @param context_overlap: Size of the overlap between two consecutive frames.
    @param positive_offset: Offset from the end of the context measured in frames, that describes the distance
            between the context and the positive sample. If the offset is 0 then the positive sample comes directly
            after the context.
    @param ns_same_segment_prob: Probability, that a negative sample will come from the same segment as
            the positive sample. If the probability is 0 then all negative samples are equally likely to come from
            every segment in the buffer. If the probability is 1 then all negative samples come from the same segment
            as the positive sample. By default, all segments have the same probability of being sampled.
    @param samples_per_patient: Number of samples before the buffer is updated.
            This is done in order to decrease the number of i/o operations.

    @return: Generator of: input data as a dictionary {context, samples}, output data as the position
    of the positive sample in the samples array. Context is an array of (optionally) overlapping frames.
    Samples is an array of frames to predict from (1 positive sample and rest negative samples).
    """
    # compute context size measured in amplitude samples, adjust for the frame overlap
    context_size = context_size * (frame_size - context_overlap)
    for patients_buffer in buffered_patient_generator:
        for _ in range(samples_per_patient):
            # collect (optionally) overlapping frames that will form the context
            # choose context start such that the positive sample will remain within the segment
            patient_index, segment_index = _choose_random_segment(patients_buffer)
            _, (signal, _) = patients_buffer[patient_index]
            segment_size = signal.shape[1]
            context_start = np.random.randint(segment_size - (context_size + frame_size * (positive_offset + 1)))
            context_end = context_start + context_size
            context = []
            for context_frame_start in range(context_start, context_end, frame_size - context_overlap):
                context_frame_end = context_frame_start + frame_size
                context_frame = signal[segment_index, context_frame_start:context_frame_end]
                context_frame = np.expand_dims(context_frame, axis=1)  # add channel dimension
                context.append(context_frame)
            context = np.array(context)
            # collect positive sample from the future relative to the context
            positive_sample_start = context_start + context_size + frame_size * positive_offset
            positive_sample_end = positive_sample_start + frame_size
            positive_sample = signal[segment_index, positive_sample_start:positive_sample_end]
            positive_sample = np.expand_dims(positive_sample, axis=1)  # add channel dimension
            # collect negative samples
            #  note that if the patient buffer contains only 1 patient then
            #  all negative samples will also come from this patient
            samples = []
            p = (patient_index, segment_index, ns_same_segment_prob) if ns_same_segment_prob else None
            ns_indices = _choose_random_segment(patients_buffer, size=ns, segment_p=p)
            for ns_patient_index, ns_segment_index in ns_indices:
                _, (ns_signal, _) = patients_buffer[ns_patient_index]
                ns_segment_size = ns_signal.shape[1]
                negative_sample_start = np.random.randint(ns_segment_size - frame_size)
                negative_sample_end = negative_sample_start + frame_size
                negative_sample = ns_signal[ns_segment_index, negative_sample_start:negative_sample_end]
                negative_sample = np.expand_dims(negative_sample, axis=1)  # add channel dimension
                samples.append(negative_sample)
            # randomly insert the positive sample among the negative samples
            # the label references the position of the positive sample among all samples
            y = np.random.randint(ns + 1)
            samples.insert(y, positive_sample)
            samples = np.array(samples)
            x = {'context': context,
                 'samples': samples}
            yield x, y


def uniform_patient_generator(db_dir, patient_ids, repeat=True, shuffle=True, include_labels=True,
                              unzipped=False):
    """
    Yield data for each patient in the array.

    @param db_dir: Database directory.
    @param patient_ids: Array of patient ids.
    @param repeat: Whether to restart the generator when the end of patient array is reached.
    @param shuffle: Whether to shuffle patient ids.
    @param include_labels: Whether patient data should also include labels or only the signal.
    @param unzipped: Whether patient files are unzipped.

    @return: Generator that yields a tuple of patient id and patient data.
    """
    if shuffle:
        patient_ids = np.copy(patient_ids)
    while True:
        if shuffle:
            np.random.shuffle(patient_ids)
        for patient_id in patient_ids:
            patient_data = load_patient_data(db_dir, patient_id, include_labels=include_labels, unzipped=unzipped)
            yield patient_id, patient_data
        if not repeat:
            break


def random_patient_generator(db_dir, patient_ids, patient_weights=None, include_labels=True,
                             unzipped=False):
    """
    Samples patient data from the provided patient distribution.

    @param db_dir: Database directory.
    @param patient_ids: Array of patient ids.
    @param patient_weights: Probabilities associated with each patient. By default assumes a uniform distribution.
    @param include_labels: Whether patient data should also include labels or only the signal.
    @param unzipped: Whether patient files are unzipped.

    @return: Generator that yields a tuple of patient id and patient data.
    """
    while True:
        for patient_id in np.random.choice(patient_ids, size=1024, p=patient_weights):
            patient_data = load_patient_data(db_dir, patient_id, include_labels=include_labels, unzipped=unzipped)
            yield patient_id, patient_data


def count_labels(labels, num_classes):
    """
    Count the number of labels in all segments.

    @param labels: Array of tuples of indices, labels. Each tuple contains the labels within a segment.
    @param num_classes: Number of classes (either beat or rhythm depending on the label type).

    @return: Numpy array of label counts of shape (num_segments, num_classes).
    """
    return np.array([
        np.bincount(segment_labels, minlength=num_classes) for _, segment_labels in labels
    ])


def calculate_durations(labels, num_classes):
    """
    Calculate the duration of each label in all segments.

    @param labels: Array of tuples of indices, labels. Each tuple corresponds to a segment.
    @param num_classes: Number of classes (either beat or rhythm depending on the label type).

    @return: Numpy array of label durations of shape (num_segments, num_classes).
    """
    num_segments = len(labels)
    durations = np.zeros((num_segments, num_classes), dtype='int32')
    for segment_index, (segment_indices, segment_labels) in enumerate(labels):
        segment_durations = np.diff(segment_indices, prepend=0)
        for label in range(num_classes):
            durations[segment_index, label] = segment_durations[segment_labels == label].sum()
    return durations


def unzip_patient_data(db_dir, patient_id, out_dir=None):
    """
    Unzip signal and labels file into the specified output directory.

    @param db_dir: Database directory.
    @param patient_id: Id of a patient.
    @param out_dir: Output directory.

    @return: None.
    """
    signal, labels = load_patient_data(db_dir, patient_id)
    out_signal_file = os.path.join(out_dir or os.path.curdir, '{:05d}_batched.npy'.format(patient_id))
    out_labels_file = os.path.join(out_dir or os.path.curdir, '{:05d}_batched_lbls.npz'.format(patient_id))
    np.save(out_signal_file, signal)
    np.savez(out_labels_file, **labels)


def load_patient_data(db_dir, patient_id, include_labels=True, unzipped=False):
    """
    Load patient data. Note, that labels are automatically flattened.

    @param db_dir: Database directory.
    @param patient_id: Id of a patient.
    @param include_labels: Whether patient data should also include labels or only the signal.
    @param unzipped: Whether patient files are unzipped.

    @return: Tuple of signal, labels.
    """
    signal = load_signal(db_dir, patient_id, unzipped=unzipped)
    if include_labels:
        labels = load_labels(db_dir, patient_id, unzipped=unzipped)
        return signal, labels
    else:
        return signal, None


def load_signal(db_dir, patient_id, unzipped=False, mmap_mode=None):
    """
    Load signal from a patient file.

    @param db_dir: Database directory.
    @param patient_id: Id of a patient.
    @param unzipped: Whether signal file is unzipped. If true then the file is treated as an unzipped numpy file.
    @param mmap_mode: Memory-mapped mode. Used in the numpy.load function.

    @return: Numpy array of shape (num_segments, segment_size).
    """
    if unzipped:
        signal = np.load(os.path.join(db_dir, '{:05d}_batched.npy'.format(patient_id)), mmap_mode=mmap_mode)
    else:
        signal = load_pkl(os.path.join(db_dir, '{:05d}_batched.pkl.gz'.format(patient_id)))
    return signal


def load_labels(db_dir, patient_id, flatten=True, unzipped=False):
    """
    Load labels from a patient file.

    @param db_dir: Database directory.
    @param patient_id: Id of a patient.
    @param flatten: Whether raw labels should be flattened.
    @param unzipped: Whether labels file is unzipped.

    @return: Raw or flattened labels.
    """
    if unzipped:
        flat_labels = np.load(os.path.join(db_dir, '{:05d}_batched_lbls.npz'.format(patient_id)), allow_pickle=True)
        return flat_labels
    else:
        raw_labels = load_pkl(os.path.join(db_dir, '{:05d}_batched_lbls.pkl.gz'.format(patient_id)))
        if flatten:
            flat_labels = flatten_raw_labels(raw_labels)
            return flat_labels
        else:
            return raw_labels


def flatten_raw_labels(raw_labels):
    """
    Flatten raw labels from a patient file for easier processing.

    @param raw_labels: Array of dictionaries containing the beat and rhythm labels for each segment.
            Note, that beat and rhythm label indices do not always overlap.

    @return: Dictionary of beat and rhythm arrays.
    Each array contains a tuple of indices, labels for each segment.
    """
    num_segments = len(raw_labels)
    labels = {'btype': [], 'rtype': [], 'size': num_segments}
    for label_type in ['btype', 'rtype']:
        for segment_labels in raw_labels:
            flat_indices = []
            flat_labels = []
            for label, indices in enumerate(segment_labels[label_type]):
                flat_indices.append(indices)
                flat_labels.append(np.repeat(label, len(indices)))
            flat_indices = np.concatenate(flat_indices)
            flat_labels = np.concatenate(flat_labels)
            sort_index = np.argsort(flat_indices)
            flat_indices = flat_indices[sort_index]
            flat_labels = flat_labels[sort_index]
            labels[label_type].append((flat_indices, flat_labels))
    return labels


def get_rhythm_durations(indices, labels=None, start=0, end=None):
    """
    Compute the durations of each rhythm within the specified frame.
    The indices are assumed to specify the end of a rhythm.

    @param indices: Array of rhythm indices. Indices are assumed to be sorted.
    @param labels: Array of rhythm labels.
    @param start: Index of the first sample in the frame.
    @param end: Index of the last sample in the frame. By default the last element in the indices array.

    @return: Tuple of: (rhythm durations, rhythm labels) in the provided frame
    or only rhythm durations if labels are not provided.
    """
    if end is None:
        end = indices[-1]
    if start >= end:
        raise ValueError('`end` must be greater than `start`')
    # find the first rhythm label after the beginning of the frame
    start_index = np.searchsorted(indices, start, side='right')
    # find the first rhythm label after or exactly at the end of the frame
    end_index = np.searchsorted(indices, end, side='left') + 1
    frame_indices = indices[start_index:end_index]
    # compute the duration of each rhythm adjusted for the beginning and end of the frame
    frame_rhythm_durations = np.diff(frame_indices[:-1], prepend=start, append=end)
    if labels is None:
        return frame_rhythm_durations
    else:
        frame_labels = labels[start_index:end_index]
        return frame_rhythm_durations, frame_labels


def get_complete_beats(indices, labels=None, start=0, end=None):
    """
    Find all complete beats within a frame i.e. start and end of the beat lie within the frame.
    The indices are assumed to specify the end of a heartbeat.

    @param indices: Array of beat indices. Indices are assumed to be sorted.
    @param labels: Array of beat labels.
    @param start: Index of the first sample in the frame.
    @param end: Index of the last sample in the frame. By default the last element in the indices array.

    @return: Tuple of: (beat indices, beat labels) in the provided frame
    or only beat indices if labels are not provided.
    """
    if end is None:
        end = indices[-1]
    if start >= end:
        raise ValueError('`end` must be greater than `start`')
    start_index = np.searchsorted(indices, start, side='left') + 1
    end_index = np.searchsorted(indices, end, side='right')
    indices_slice = indices[start_index:end_index]
    if labels is None:
        return indices_slice
    else:
        label_slice = labels[start_index:end_index]
        return indices_slice, label_slice


def get_rhythm_label(durations, labels):
    """
    Determine rhythm label based on the longest rhythm among undefined / afib / aflut if present,
    otherwise the longer among end / noise / normal.

    @param durations: Array of rhythm durations
    @param labels: Array of rhythm labels.

    @return: Rhythm label as an integer.
    """
    # sum up the durations of each rhythm
    summed_durations = np.zeros(len(ds_rhythm_names))
    for label in ds_rhythm_names:
        summed_durations[label] = durations[labels == label].sum()
    longest_hp_rhythm = np.argmax(summed_durations[_HI_PRIO_RHYTHMS])
    if summed_durations[_HI_PRIO_RHYTHMS][longest_hp_rhythm] > 0:
        y = _HI_PRIO_RHYTHMS[longest_hp_rhythm]
    else:
        longest_lp_rhythm = np.argmax(summed_durations[_LO_PRIO_RHYTHMS])
        # handle the case of no detected rhythm
        if summed_durations[_LO_PRIO_RHYTHMS][longest_lp_rhythm] > 0:
            y = _LO_PRIO_RHYTHMS[longest_lp_rhythm]
        else:
            y = 0  # undefined rhythm
    return y


def get_beat_label(labels):
    """
    Determine beat label based on the occurrence of pac / abberated / pvc,
    otherwise pick the most common beat type among the normal / undefined.

    @param labels: Array of beat labels.

    @return: Beat label as an integer.
    """
    # calculate the count of each beat type in the frame
    beat_counts = np.bincount(labels, minlength=len(ds_beat_names))
    most_hp_beats = np.argmax(beat_counts[_HI_PRIO_BEATS])
    if beat_counts[_HI_PRIO_BEATS][most_hp_beats] > 0:
        y = _HI_PRIO_BEATS[most_hp_beats]
    else:
        most_lp_beats = np.argmax(beat_counts[_LO_PRIO_BEATS])
        # handle the case of no detected beats
        if beat_counts[_LO_PRIO_BEATS][most_lp_beats] > 0:
            y = _LO_PRIO_BEATS[most_lp_beats]
        else:
            y = 0  # undefined beat
    return y


def get_heart_rate_label(qrs_indices, fs=None):
    """
    Determine the heart rate label based on an array of QRS indices (separating individual heartbeats).
    The QRS indices are assumed to be measured in seconds if sampling frequency `fs` is not specified.
    The heartbeat label is based on the following BPM (beats per minute) values: (0) tachycardia <60 BPM,
    (1) bradycardia >100 BPM, (2) healthy 60-100 BPM, (3) noisy if QRS detection failed.

    @param qrs_indices: Array of QRS indices.
    @param fs: Sampling frequency of the signal.

    @return: Heart rate label as an integer.
    """
    if len(qrs_indices) > 1:
        rr_intervals = np.diff(qrs_indices)
        if fs is not None:
            rr_intervals = rr_intervals / fs
        bpm = 60 / rr_intervals.mean()
        if bpm < 60:
            return _HR_BRADYCARDIA
        elif bpm <= 100:
            return _HR_NORMAL
        else:
            return _HR_TACHYCARDIA
    else:
        return _HR_NOISE


def normalize(array, inplace=False):
    """
    Normalize an array using the mean and standard deviation calculated over the entire dataset.

    @param array: Numpy array to normalize.
    @param inplace: Whether to perform the normalization steps in-place.

    @return: Normalized array.
    """
    if inplace:
        array -= ds_mean
        array /= ds_std
    else:
        array = (array - ds_mean) / ds_std
    return array


def _choose_random_segment(patients, size=None, segment_p=None):
    """
    Choose a random segment from an array of patient data. Each segment has the same probability of being chosen.
    Probability of a single segment may be changed by passing the `segment_p` argument.

    @param patients: An array of tuples of patient id and patient data.
    @param size: Number of the returned random segments. Defaults to 1 returned random segment.
    @param segment_p: Fixed probability of a chosen segment. `segment_p` should be a tuple of:
            (patient_index, segment_index, segment_probability)

    @return: One or more tuples (patient_index, segment_index) describing the randomly sampled
    segments from the patients buffer.
    """
    num_segments_per_patient = np.array([signal.shape[0] for _, (signal, _) in patients])
    first_segment_index_by_patient = np.cumsum(num_segments_per_patient) - num_segments_per_patient
    num_segments = num_segments_per_patient.sum()
    if segment_p is None:
        p = np.ones(num_segments) / num_segments
    else:
        patient_index, segment_index, segment_prob = segment_p
        p_index = first_segment_index_by_patient[patient_index] + segment_index
        if num_segments <= p_index < 0:
            raise ValueError('The provided patient and segment indices are invalid')
        if 1. < segment_prob < 0.:
            raise ValueError('Probability must lie in the [0, 1] interval')
        p = (1 - segment_prob) * np.ones(num_segments) / (num_segments - 1)
        p[p_index] = segment_prob
    segment_ids = np.random.choice(num_segments, size=size, p=p)
    if size is None:
        patient_index = np.searchsorted(first_segment_index_by_patient, segment_ids, side='right') - 1
        segment_index = segment_ids - first_segment_index_by_patient[patient_index]
        return patient_index, segment_index
    else:
        indices = []
        for segment_id in segment_ids:
            patient_index = np.searchsorted(first_segment_index_by_patient, segment_id, side='right') - 1
            segment_index = segment_id - first_segment_index_by_patient[patient_index]
            indices.append((patient_index, segment_index))
        return indices
