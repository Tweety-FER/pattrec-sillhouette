import numpy as np
import cv2
import time
import copy
import Classify
from os import listdir
from os.path import join
from operator import itemgetter
from sys import argv

g_frames = None

# Chain code directions
chains = {
    (1, 0) : 0,
    (1, 1) : 1,
    (0, 1) : 2,
    (-1, 1) : 3,
    (-1, 0) : 4,
    (-1, -1) : 5,
    (0, -1) : 6,
    (1, -1) : 7
}

def get_chain_vector(contours):
    """Transforms given contours into a 8-dimensional normalised vector.
    Each dimension of the vector corresponds to one of the 8 dimensions of the chain code.

    contours : Countours from an image
    returns: Normalised vector representation of contours
    """
    vec = get_chain_code(contours)
    if not vec.any():
        return np.array([])

    vec = vectorize_chain_code(vec, 8)
    return vec / np.linalg.norm(vec) - np.array([1.0 / 8] * 8)

def get_chain_code(contours):
    """Transforms given contours into cyclic chain code, where shifts between two points are 
    represented by their direction (see variable chains).

    contours : The contours of an image
    returns : Chain code representation of contours
    """
    #Empty
    if not contours:
        return np.array([], dtype=np.int32)


    chain = []
    for contour in contours:
        contour = contour + [contour[0]]
        clen = len(contour)


        for i in xrange(1, clen):
            diff = contour[i][0] - contour[i - 1][0]
            diff = diff[0], diff[1]
            chain.append(chains[diff])

    return np.array(chain, dtype=np.int32)

def vectorize_chain_code(code, k = 8):
    """Transforms a given chain code into a k-dimensional vector, where each 
    dimensions is the count of that number appearing in the chain code.

    code : Chain code representation of contour(s)
    k : Dimensionality of output vector
    returns: A vector representation of the chain code, where each dimension's magnitude
    is the number of appearances of that chain code element in chain code
    """
    return np.array(np.bincount(code, minlength=k)[:k], dtype=np.float32)


def process_frames(source, display = False, clear_iters = 1, blur_kernel = (5, 5), max_frames = -1):
    """Reads a video from source (a file or a camera) and performs preprocessing and transformation into
    a vector representation for each frame. The preprocessing consists of blurring, eroding, dilating, extracting
    the contour, then transforming it into a vector representation of its chain code form. See function get_chain_vector.

    source : Video source. File path or 0 for camera.
    display : Show contour and print progress
    clear_iters : How many iterations of erosion od dilation to perform
    blur_kernel : Kernel for Gaussian blur
    max_frames : Number of frames to read. -1 for all. Needs to be used with camera to prevent infinite filming.
    returns : Vector representations of motion in frames in video
    """
    kernel = np.ones((3, 3), np.uint8)

    cap = cv2.VideoCapture(source)
    fgbg = cv2.BackgroundSubtractorMOG(60 * 30, 256, 0.9, 0.01)
    frames = []

    ret, frame = cap.read()

    if not ret:
        return []

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(frame, blur_kernel, 0)
    erosion = cv2.erode(blur, kernel, iterations = clear_iters)
    dilatation = cv2.dilate(erosion, kernel, iterations = clear_iters)

    avg = np.float32(dilatation)

    frame_cnt = 0

    while(1):
        ret, frame = cap.read()
		
        if not ret: break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(frame, (5, 5), 0)
        erosion = cv2.erode(blur, kernel, iterations = clear_iters)
        dilatation = cv2.dilate(erosion, kernel, iterations = clear_iters)

        cv2.accumulateWeighted(dilatation, avg, 0.05)
        frames.append(dilatation)

        frame_cnt += 1
        if max_frames != -1 and frame_cnt >= max_frames:
            break

    cap.release()

    average = cv2.convertScaleAbs(avg)
    fgbg.apply(average)

    chains = []

    for frame in frames:
        img = fgbg.apply(frame)
        img1 = copy.deepcopy(img)
        contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        chains.append(get_chain_vector(contours))

        if display == True and contours:
            cv2.drawContours(img1, contours, -1, (255,0,0), 1)
            cv2.imshow('Silueta', img1)
            cv2.waitKey(1)

    
    cv2.destroyAllWindows()

    return [frame for frame in chains if frame.any()]

def train_from_files(root, outdir='train.data', separator = ' ', display = False):
    """Gets clustering centroids for 

    """
    labels = []
    centroids = []
    for class_dir in listdir(root):
        if display: print "Processing", class_dir

        labels.append(class_dir)
        vectors = []
        dir_path = join(root, class_dir)

        for ex_file in listdir(dir_path):
            vectors += process_frames(join(dir_path, ex_file))

        centroids.append(Classify.centroid(*vectors))

    if display: print 'Storing data...'

    with open(outdir, 'w') as fp:
        for i in xrange(len(labels)):
            for cen in centroids[i]:
                fp.write(`cen` + separator)
            fp.write(labels[i] + '\n')

    if display: print 'Done training'

def test_from_file(root, display = False, use_smoothing = False):
    guessed = []
    real = []

    for t_file in listdir(root):
        action_real = [filter(lambda x: x.isalpha(), t_file.split('.')[0].split('_')[1])]
        action_guessed = classify(join(root, t_file), display = display, use_smoothing = use_smoothing)
        action_real = action_real * len(action_guessed)

        real += action_real
        guessed += action_guessed

    print guessed
    print real

    return np.float32(sum([1 if guessed[i] == real[i] else 0 for i in xrange(len(real))])) / len(real)


def load_classes_from_file(fname, separator=' '):
    data = np.loadtxt(fname, delimiter = separator, dtype = str)
    labels = data[:,-1]
    centroids = np.array(data[:,:-1], dtype = np.float32)

    return centroids, labels

def classify(fname, classes_file = 'train.data', display = False, metric = Classify.norm_angle_dist, max_frames = -1, use_smoothing = False):
    global g_frames
    if display:
        if fname != 0:
            print 'Processing', fname
        else:
            print 'Using the camera'

    class_centr, class_labels = load_classes_from_file(classes_file)
    k = np.size(class_labels)

    frames = process_frames(fname, display = display, max_frames = max_frames)

    final = do_classify(frames, class_labels, class_centr)

    if use_smoothing:
        fixed_frames = smooth(final, frames)
        g_frames = fixed_frames
        return do_classify(fixed_frames, class_labels, class_centr)

    else:
        g_frames = frames

    return final

def do_classify(frames, class_labels, class_centr):
    k = np.size(class_labels)
    labels, _, centrs = Classify.cluster(frames, k)

    real_labels = []

    for i in xrange(k):
        c = centrs[i]
        d = None
        real_i = 0

        for j in xrange(k):
            cc = class_centr[j]
            cd = np.linalg.norm(c - cc)#metric(c, cc)
            if d == None or cd < d:
                d = cd
                real_i = j

        real_labels.append(real_i)

    return [class_labels[real_labels[i]] for i in labels]


def most_common_frame(frame_labels):
    d = {}
    for l in frame_labels: 
        d[l] = d.get(l, 0) + 1

    return max(d.iteritems(), key = itemgetter(1))[0]

def smooth(labels, frames, window = 1, K=2, weights = np.array([1.0, 1.0, 0.0, 1.0, 1.0])):
    assert len(weights) == 2 * K + 1, 'Weights must be of length 2 * K + 1'

    len_labels = len(labels)
    len_frames = len(frames)
    marked = []
    for i in xrange(len_labels):
        if any([labels[j] != labels[i] for j in xrange(max(0, i - window), min(len_labels, i + window + 1))]):
            marked.append(i)


    for i in marked:
        correction = np.zeros(frames[i].shape)
        index = 0
        for n in xrange(max(0, i - K), min(len_frames, i + K + 1)):
            correction += weights[index] * frames[i]
            index += 1
        correction /= 2 * K

        frames[i] = correction

    return frames


if __name__ == '__main__':
    argv = argv[1:]
    action = argv[0].lower()

    if action == 'train':
        print "Training"
        train_from_files(argv[1], display = True)
    elif action == 'test':
        print test_from_file(argv[1], display = True, use_smoothing = True)
    elif action == 'classify':
        source = 0 if argv[1] == '0' else argv[1]
        max_frames = int(argv[2]) if len(argv) > 2 else -1
        labels = classify(source, display = True, max_frames = max_frames, use_smoothing = True)
        print labels
        print "The action in this video is", most_common_frame(labels)
    else:
        print action, "is an invalid command"
        print "Valid commands are:"
        print "\ttrain train_dir"
        print "\ttest test_dir"
        print "\tclassify source [max_frames]"
        print "Where source can be a path or 0 for camera and max_frames has to be used with a camera"
    
