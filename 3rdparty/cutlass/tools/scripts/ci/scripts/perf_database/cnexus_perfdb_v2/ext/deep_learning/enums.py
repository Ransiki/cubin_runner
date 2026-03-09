from ...base import LooseEnum as Enum


class Network(Enum):
    """
    Network name.

    A network is considered different from each other if there are topolological
    differences between the networks. E.g. Jasper 10x3 is different from Jasper
    10x5 for this reason.

    """

    # Public API
    _2D_UNet = "2D U-Net"  #: Same as the 3D UNet, but BatchNormalization nodes are replaced with Add nodes
    _3D_UNet = "3D U-Net"  #:
    _3D_UNet_Deconvolution = "3D U-Net (With Deconvolution Layers)"  #:
    _3D_UNet_Upsampling = "3D U-Net (With Upsampling)"  #:
    AlexNet_v1 = "AlexNet v1"  #:
    Artifact_Removal = "Artifact Removal"  #:
    BART = "BART"  #:
    BERT_Base = "BERT-Base"  #:
    BERT_Large = "BERT-Large"  #:
    Deep_ASR = "Deep ASR"  #:
    CosmoFlow = "CosmoFlow"  #:
    DLRM = "DLRM"  #:
    Electra = "Electra"  #:
    EfficientDet = "EfficientDet"  #:
    EfficientNet = "EfficientNet"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B0 = "EfficientNet-B0"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B1 = "EfficientNet-B1"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B2 = "EfficientNet-B2"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B3 = "EfficientNet-B3"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B4 = "EfficientNet-B4"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B5 = "EfficientNet-B5"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B6 = "EfficientNet-B6"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B7 = "EfficientNet-B7"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    EfficientNet_B8 = "EfficientNet-B8"  #: Example Implementation + Explanation: https://github.com/lukemelas/EfficientNet-PyTorch
    FastPitch = "FastPitch"  #:
    Faster_R_CNN = "Faster R-CNN"  #: https://arxiv.org/abs/1506.01497
    FCN_MobileNet_v1 = "FCN-MobileNet v1"  #: https://arxiv.org/abs/1704.04861
    FCN_MobileNet_v2 = "FCN-MobileNet v2"  #:
    FCN_ResNet_18 = "FCN-ResNet-18"  #:
    GNMT = "GNMT"  #:
    HiFi_GAN = "HiFi-GAN"  #:
    Inception_v1 = "Inception v1"  #: Also known as GoogleNet
    Inception_v1_Batch_Normalization = "Inception v1 (Batch Normalization)"  #: Also known as GoogleNet, with batch normalization (https://en.wikipedia.org/wiki/Batch_normalization).
    Inception_v2 = "Inception v2"  #:
    Inception_v3 = "Inception v3"  #:
    Inception_v4 = "Inception v4"  #:
    Jasper_10x3 = "Jasper 10x3"  #:
    Jasper_10x5 = "Jasper 10x5"  #:
    LSTM = "LSTM"  #:
    LSTM_Peephole = (
        "LSTM Peephole"  #: https://machinelearning.wtf/terms/peephole-connection-lstm/
    )
    LSTM_Projection = (
        "LSTM Projection"  #: https://www.hindawi.com/journals/jr/2017/2061827/
    )
    Mask_R_CNN = "Mask R-CNN"  #:
    Microsoft_Cortana = "Microsoft Cortana"  #:
    Microsoft_Cortana_Bunk = "Microsoft Cortana (Bunk Variation)"  #:
    Microsoft_Cortana_Ragged = "Microsoft Cortana (Ragged Variation)"  #:
    Microsoft_Cortana_Sunk = "Microsoft Cortana (Sunk Variation)"  #:
    Microsoft_Cortana_Tiny = "Microsoft Cortana (Tiny Variation)"  #:
    MobileNet_v1 = "MobileNet v1"  #:
    MobileNet_v2 = "MobileNet v2"  #:
    NCF = "NCF"  #:
    NCF_Amazon = (
        "NCF Amazon"  #: https://confluence.nvidia.com/display/DL/NCF+(MLP)+Perf+Survey
    )
    NCF_Apple = (
        "NCF Apple"  #: https://confluence.nvidia.com/display/DL/NCF+(MLP)+Perf+Survey
    )
    NCF_Bytedance = "NCF Bytedance"  #: https://confluence.nvidia.com/display/DL/NCF+(MLP)+Perf+Survey
    NCF_Deep_Recommender = "NCF DeepRecommender"  #: https://confluence.nvidia.com/display/DL/NCF+(MLP)+Perf+Survey
    NCF_Ebay = (
        "NCF Ebay"  #: https://confluence.nvidia.com/display/DL/NCF+(MLP)+Perf+Survey
    )
    NCF_Facebook_10L_P1 = "NCF Facebook (10L)"  #: Facebook NCF with 10 Layers and 1 tensor input https://confluence.nvidia.com/pages/viewpage.action?pageId=172467135
    NCF_Facebook_4L_P1 = "NCF Facebook (4L)"  #: Facebook NCF with 4 Layers and 1 tensor input https://confluence.nvidia.com/pages/viewpage.action?pageId=172467135
    NCF_Facebook_4L_P50 = "NCF Facebook (4L P50)"  #: Facebook NCF with 4 Layers and 50 tensor inputs https://confluence.nvidia.com/pages/viewpage.action?pageId=172467135
    NCF_SAP = (
        "NCF SAP"  #: https://confluence.nvidia.com/display/DL/NCF+(MLP)+Perf+Survey
    )
    OctConv = "OctConv"  #:
    ResNet_152 = "ResNet-152"  #: Released in 2015 by Microsoft Research Asia, the ResNet architecture (with its three realizations ResNet-50, ResNet-101 and ResNet-152) obtained very successful results in the ImageNet and MS-COCO competition. The core idea exploited in these models, residual connections, is found to greatly improve gradient flow, thus allowing training of much deeper models with tens or even hundreds of layers. Source: https://resources.wolframcloud.com/NeuralNetRepository/resources/ResNet-152-Trained-on-ImageNet-Competition-Data
    ResNet_18 = "ResNet-18"  #:
    ResNet_34 = "ResNet-34"  #:
    ResNet_50_v1 = "ResNet-50 v1"  #:
    ResNet_50_v1_5 = "ResNet-50 v1.5"  #:
    ResNet_50_v2 = "ResNet-50 v2"  #:
    ResNeXt_101 = "ResNeXt-101"  #: https://arxiv.org/abs/1611.05431
    ResNeXt_101_32_4 = "ResNeXt-101 32x4"  #:
    ResNeXt_152 = "ResNeXt-152"  #: https://github.com/cypw/ResNeXt-1
    ResNeXt_50 = "ResNeXt-50"  #:
    RNNT = "RNN-T"  #:
    SE_ResNeXt_101_32_4 = "SE ResNeXt-101 32x4"  #:
    SSD = "SSD"  #:
    SSD_Large = "SSD Large"  #:
    SSD_MobileNet = "SSD-MobileNet"  #:
    SSD_ResNet_34_300 = "SSD-ResNet-34 300"  #: https://developers.arcgis.com/python/guide/how-ssd-works/
    SSD_Small = "SSD Small"  #:
    SSD_v1_1 = "SSD v1.1"  #:
    SSD_v1_2 = "SSD v1.2"  #:
    Super_Resolution = "Super Resolution"  #:
    Tacotron = "Tacotron"  #:
    Tacotron_v2 = (
        "Tacotron v2"  #: https://github.com/NVIDIA/TensorRT/tree/master/demo/Tacotron2
    )
    Tacotron_v2_Decoder_Iter = "Tacotron v2 Decoder (Iterated Variation)"  #: Tacotron v2 is split into 3 stages. This is stage 2/3. This stage is data dependent is typically run multiple times in an outside loop. This is done because the decoder loop limit is data-dependent and does not export successfully using torch JIT export.
    Tacotron_v2_Decoder_Looped = "Tacotron v2 Decoder (Loop Variation)"  #: Tacotron v2 is split into 3 stages. This is stage 2/3, where a loop is encoded into the network itself. This is done because the decoder loop limit is data-dependent and does not export successfully using torch JIT export.
    Tacotron_v2_Encoder = "Tacotron v2 Encoder"  #: Tacotron v2 is split into 3 stages. This is stage 1/3. This is done because the decoder loop limit is data-dependent and does not export successfully using torch JIT export.
    Tacotron_v2_PostNet = "Tacotron v2 PostNet"  #: Tacotron v2 is split into 3 stages. This is stage 3/3. This is done because the decoder loop limit is data-dependent and does not export successfully using torch JIT export.
    Transformer = "Transformer"  #:
    Transformer_XL = "Transformer XL"  #:
    UNet_Industrial = "U-Net Industrial"  #:
    UNet_Medical = "U-Net Medical"  #:
    VGG_16 = "VGG-16"  #:
    VNet_v1 = "V-Net v1"  #:
    VNet_v2 = "V-Net v2"  #:
    Waveglow = "Waveglow"  #:
    WaveRNN = "WaveRNN"  #:
    WideAndDeep = "Wide & Deep"  #:
    Xception = "Xception"  #:
    Yolo_v1 = "Yolo v1"  #:
    Yolo_v2 = "Yolo v2"  #:
    Yolo_v3 = "Yolo v3"  #:


class NetworkVariant(Enum):
    """
    Network variant

    Multiple clients may implement the same network from a topologlical
    perspective, but may yield slightly different networks in other regards.
    Recording the variant by the client records these slight differences.
    """

    # Public API
    JOC = "JOC"  #: Joy of Cooking
    MLPerf = "MLPerf"  #:


class NetworkApplication(Enum):
    """
    Network Application

    The task that a network is meant to perform. All computer vision tasks are collected under 'Vision'
    """

    # Public API
    Vision = "Vision"  #:
    Recsys = "Recommendation System"  #: Recommendation System
    ASR = "Automated Speech Recognition"  #: Automated Speech Recognition
    TTS = "Text to Speech"  #: Text To Speech
    NLP = "Natural Language Processing"  #: Natural Language Processing


class OperatorType(Enum):
    """
    Operator type

    Reference:
    https://confluence.nvidia.com/display/GCA/DNNX+-+DNN+Description+Format#DNNX-DNNDescriptionFormat-Operators.1
    https://confluence.nvidia.com/display/GCA/Proposed+DNNX+Operators
    """

    Accuracy = "Accuracy"  #:
    BatchNorm = "BatchNorm"  #:
    Bias = "Bias"  #:
    Convolution = "Convolution"  #:
    Deconvolution = "Deconvolution"  #:
    dropdout = "dropout"  #:
    EltWise = "EltWise"  #:
    FullyConnected = "FullyConnected"  #:
    LRN = "LRN"  #:
    Pooling = "Pooling"  #:
    PReLU = "PReLU"  #:
    Recurrent = "Recurrent"  #:
    RecurrentBegin = "RecurrentBegin"  #:
    RecurrentEnd = "RecurrentEnd"  #:
    RecurrentMacro = "RecurrentMacro"  #:
    Reduction = "Reduction"  #:
    ReLU = "ReLU"  #:
    Scale = "Scale"  #:
    Softmax = "Softmax"  #:
    SoftmaxWithLoss = "SoftmaxWithLoss"  #:


class ImplementationType(Enum):
    """
    Implementation type

    Reference: https://confluence.nvidia.com/display/GCA/DNN+Description+Format#DNNDescriptionFormat-OperatorImplementations.1
    """

    Analytical = "Analytical"  #:
    CublasTest = "CublasTest"  #:
    CudnnTest = "CudnnTest"  #:
    DLSim = "DLSim"  #:
    FastKernels = "FastKernels"  #:


class Instruction(Enum):

    # Public API
    ffma = "ffma"  #:
    hfma = "hfma"  #:
    idp4 = "idp4"  #:
    hmma = "hmma"  #:
    hmma_sp = "hmma.sp"  #:
    imma = "imma"  #:
    imma_sp = "imma.sp"  #:


class Mode(Enum):
    """Run mode: either `inference` or `training`"""

    # Public API
    Inference = "inference"  #:
    Training = "training"  #:


class Framework(Enum):
    """List of supported frameworks"""

    # Public API
    Caffe = "Caffe"  #:
    MXNet = "MXNet"  #:
    PyTorch = "PyTorch"  #:
    TensorFlow = "TensorFlow"  #:
    TensorFlow2 = "TensorFlow 2"  #:
    TensorRT = "TensorRT"  #:


class NetworkDescriptionFormat(Enum):
    """A specific "save format" of a network.

    Has a mild effect on performance.
    """

    Caffe_Prototxt = "Caffe Prototxt"  #: http://caffe.berkeleyvision.org/
    ONNX = "ONNX"  #:
    Protobuf = "Protobuf"  #:
    TensorRT_Engine = "TensorRT Engine"  #: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#serial_model_python
    UFF = "Universal Framework Format"  #:


class Platform(Enum):
    """
    Platform

    Reference: https://confluence.nvidia.com/display/GCA/DNNX+-+DNN+Description+Format#DNNX-DNNDescriptionFormat-ProfilingCompatibility
    """

    analytical_calibrated = "analytical_calibrated"  #:
    analytical_maxopt = "analytical_maxopt"  #:
    analytical_proj = "analytical_proj"  #:
    analytical_sol = "analytical_sol"  #:
    dlsim_calibrated = "dlsim_calibrated"  #:
    dlsim_maxopt = "dlsim_maxopt"  #:
    dlsim_proj = "dlsim_proj"  #:
    dlsim_sol = "dlsim_sol"  #:
    dlsim_proj_realistic_updated = "dlsim_proj_realistic_updated"  #:
    perf_inspector = "perf_inspector"  #:
    perfalyze = "perfalyze"  #:
    perfsim = "perfsim"  #:
    silicon = "silicon"  #:


class Precision(Enum):
    """
    The format/representation of numeric data that could refer to the storage
    and computation in neural networks.

    Further Information: https://docs.google.com/document/d/1V9PQDXsgjMZOpQmfI-cTry3GcemaM2e47mUh6s1TJVk
    """

    bf16 = "bf16"
    """
    A non-standard 16-bit floating point representation that has all of the
    performance benefits of FP16 with arguably nicer practical behavior in the
    context of deep learning applications.

    This format leverages 8 exponent bits and 7 mantissa bits.
    """

    fp16 = "fp16"
    """
    The IEEE standard 16-bit floating point format. This format leverages 5
    exponent bits and 10 mantissa bits (and 1 sign bit).
    """

    fp16_bf16 = "fp16_bf16"  #: Legacy: The equivalent of bf16

    fp16_e8m7 = "fp16_e8m7"  # Legacy: The equivalent of bf16

    fp19_e8m10 = "fp19_e8m10"  # Legacy: The equivalent of tf32

    fp19_tf32 = "fp19_tf32"  #: Legacy: The equivalent of tf32

    fp32 = "fp32"
    """
    The IEEE standard 32-bit floating point format. Leverages 8 bits for the
    exponent, 23 bits for the mantissa (and 1 bit for the sign).
    """

    fp64 = "fp64"
    """
    The IEEE standard 64-bit floating point format. Leverages 11 bits for the
    exponent, 52 bits for the mantissa (and 1 bit for the sign).
    """

    int16 = "int16"
    """
    The IEEE standard 16 bit signed integer format. Leverages a two's
    complement system.
    """

    int32 = "int32"
    """
    The IEEE standard 32 bit signed integer format. Leverages a two's
    complement system.
    """

    int64 = "int64"
    """
    The IEEE standard 64 bit signed integer format. Leverages a two's
    complement system.
    """

    int8 = "int8"
    """
    The IEEE standard 8 bit signed integer format. Leverages a two's
    complement system.
    """

    tf32 = "tf32"
    """
    A new format supported by Ampere and later TensorCores that provides
    benefits of reduced precision performance without noticeably impacting
    training or inference accuracy.

    Leverages 8 bits for the exponent, 10 bits for the mantissa (and 1 bit for
    the sign), the rest of the bits are unused, but are there for hardware
    alignment reasons.
    """

    uint16 = "uint16"
    """
    The IEEE standard 16 bit unsigned integer format. Leverages a two's
    complement system.
    """

    uint32 = "uint32"
    """
    The IEEE standard 32 bit unsigned integer format. Leverages a two's
    complement system.
    """

    uint64 = "uint64"
    """
    The IEEE standard 64 bit unsigned integer format. Leverages a two's
    complement system.
    """

    uint8 = "uint8"
    """
    The IEEE standard 8 bit unsigned integer format. Leverages a two's
    complement system.
    """


class Phase(Enum):

    # Public API
    bprop = "bprop"  #:
    dgrad = "dgrad"  #:
    fprop = "fprop"  #:
    wgrad = "wgrad"  #:


class Library(Enum):
    """
    Software libraries.

    References:
    https://developer.nvidia.com/gpu-accelerated-libraries
    """

    # Public API
    AmgX = "AmgX"  #:
    ArrayFire = "ArrayFire"  #:
    CASK = "CASK"  #:
    CHOLMOD = "CHOLMOD"  #:
    cuBLAS = "cuBLAS"  #:
    CUDA = "CUDA"  #:
    CUDA_Driver = "CUDA Driver"  #:
    CUDA_Math_Library = "CUDA Math Library"  #:
    cuDNN = "cuDNN"  #:
    cuFFT = "cuFFT"  #:
    cuRAND = "cuRAND"  #:
    cuSOLVER = "cuSOLVER"  #:
    cuTENSOR = "cuTENSOR"  #:
    CUVlib = "CUVlib"  #:
    DALI = "DALI"  #:
    DeepStream_SDK = "DeepStream SDK"  #:
    FFmpeg = "FFmpeg"  #:
    Gunrock = "Gunrock"  #:
    IMSL_Numeric_Libraries = "IMSL Numeric Libraries"  #:
    Jarvis = "Jarvis"  #:
    MAGMA = "MAGMA"  #:
    MLPerf = "MLPerf"
    MXNet = "MXNet"  #:
    NVCC = "NVCC"  #:
    NVCCL = "NVCCL"  #:
    nvJPEG = "nvJPEG"  #:
    NVSHMEM = "NVSHMEM"  #:
    OpenCV = "OpenCV"  #:
    PyTorch = "PyTorch"  #:
    TensorFlow = "TensorFlow"  #:
    TensorRT = "TensorRT"  #:
    TensorRT_Siphon = "TensorRT Siphon"  #: Version number of data pull tool
    Thrust = "Thrust"  #:
    Triton = "Triton"  #:
    Triton_Ocean_SDK = "Triton Ocean SDK"  #:


class TensorLayout(Enum):
    """Tensor layout."""

    KCRS = "KCRS"  #:
    KRSC = "KRSC"  #:
    N = "N"  #:
    N32 = "N32"  #:
    N64 = "N64"  #:
    NCHW = "NCHW"  #:
    NCXHW16 = "NCXHW16"  #:
    NCXHW32 = "NCXHW32"  #:
    NCXHW4 = "NCXHW4"  #:
    NCXHW64 = "NCXHW64"  #:
    NCXHW8 = "NCXHW8"  #:
    NGHWC = "NGHWC"  #:
    NHWC = "NHWC"  #:
    T = "T"  #:
    T32 = "T32"  #:
    T64 = "T64"  #:


class GemmAlgorithm(Enum):
    """GEMM algorithm"""

    gemm = "GEMM"
    sparse_gemm = "Sparse GEMM"
    implicit_gemm = "Implicit GEMM"
    implicit_gemm_indexed = "Implicit GEMM Indexed"
    implicit_gemm_indexed_wo_smem = "Implicit GEMM Indexed WO SMEM"
    implicit_gemm_2d_tile = "Implicit GEMM 2D Tile"
    sparse_conv = "Sparse Conv"
    warp_specialized_gemm = "Warp Specialized GEMM"
    warp_specialized_implicit_gemm = "Warp Specialized Implicit GEMM"
    warp_specialized_implicit_gemm_indexed = "Warp Specialized Implicit GEMM Indexed"


class ConvolutionAlgorithm(Enum):
    """Convolution algorithm"""

    implicit_gemm = "Implicit GEMM"
    direct_convolution = "Direct Convolution"
    winograd = "Winograd"
    fft = "FFT"
