import tensorrt as trt
import sys, time
import argparse
import inspect
from pathlib import Path
from typing import *


parser = argparse.ArgumentParser(description='https://github.com/jason-li-831202/Vehicle-CV-ADAS')
parser.add_argument('--input_onnx_model', '-i', default="./ObjectDetector/models/yolov10n-coco_fp32.onnx", type=str, help='Onnx model path.')
parser.add_argument('--output_trt_model', '-o', default="./ObjectDetector/models/yolov10n-coco_fp16.trt", type=str, help='Tensorrt model path.')
parser.add_argument('--verbose', action='store_true', default=False, help='TensorRT: verbose log')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """
    def __init__(self, verbose: bool = False, workspace_size: int = 1):
        self.logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.logger.min_severity = trt.Logger.Severity.VERBOSE
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        print(self.colorstr("👉 Starting export with TensorRT Version: "), trt.__version__)
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))  # Set workspace size

        self.network = None
        self.parser = None

    def create_network(self, onnx_model_path: str):
        assert Path(onnx_model_path).exists(), print(self.colorstr("red", f"File=[{onnx_model_path}] does not exist. Please check it!"))
        EXPLICIT_BATCH = []
        if trt.__version__[0] >= '7':
            EXPLICIT_BATCH.append(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(*EXPLICIT_BATCH)
        self.parser = trt.OnnxParser(self.network, self.logger)

        print(self.colorstr('cyan', '👉 Loading the ONNX file...'))
        with open(onnx_model_path, 'rb') as f:
            if not self.parser.parse(f.read()):
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                raise RuntimeError(f'Failed to load ONNX file: {onnx_model_path}')

        print(self.colorstr('magenta', "*" * 40))
        print(self.colorstr('magenta', 'underline', '❄️  Network Description: ❄️'))
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        print(self.colorstr('bright_magenta', " - Input Info"))
        for inp in inputs:
            print(self.colorstr('bright_magenta', f'   Input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}'))
        print(self.colorstr('bright_magenta', " - Output Info"))
        for out in outputs:
            print(self.colorstr('bright_magenta', f'   Output "{out.name}" with shape {out.shape} and dtype {out.dtype}'))

    def create_engine(self, trt_model_path: str):
        start = time.time()
        inp = self.network.get_input(0)
        print(f' Note: building FP{16 if (self.builder.platform_has_fast_fp16 and inp.dtype == trt.DataType.HALF) else 32} engine as {Path(trt_model_path).resolve()}')

        if self.builder.platform_has_fast_fp16 and inp.dtype == trt.DataType.HALF:
            self.config.set_flag(trt.BuilderFlag.FP16)

        print(self.colorstr('magenta', "*" * 40))
        print(self.colorstr('👉 Building the TensorRT engine. This might take a while...'))
        
        serialized_engine = self.builder.build_serialized_network(self.network, self.config)
        if serialized_engine:
            print(self.colorstr('👉 Successfully created the serialized engine.'))

        try:
            with open(trt_model_path, 'wb') as f:
                f.write(serialized_engine)
        except Exception as e:
            print(self.colorstr('red', f'Export failure ❌ : {e}'))

        convert_time = time.time() - start
        print(self.colorstr(f'\nExport complete ✅ {convert_time:.1f}s'
                            f"\nResults saved to [{trt_model_path}]"
                            f"\nModel size:      {file_size(trt_model_path):.1f} MB"
                            f'\nVisualize:       https://netron.app'))

    @staticmethod
    def colorstr(*input):
        # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code
        *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
        colors = {
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'bright_black': '\033[90m',
            'bright_red': '\033[91m',
            'bright_green': '\033[92m',
            'bright_yellow': '\033[93m',
            'bright_blue': '\033[94m',
            'bright_magenta': '\033[95m',
            'bright_cyan': '\033[96m',
            'bright_white': '\033[97m',
            'end': '\033[0m',
            'bold': '\033[1m',
            'underline': '\033[4m'
        }
        return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def file_size(path: str):
    # Return file/dir size (MB)
    mb = 1 << 20
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / mb
    else:
        return 0.0


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    print(EngineBuilder.colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))


if __name__ == '__main__':
    args = parser.parse_args()
    print_args(vars(args))

    onnx_model_path = args.input_onnx_model
    trt_model_path = args.output_trt_model
    verbose = args.verbose

    builder = EngineBuilder(verbose)
    builder.create_network(onnx_model_path)
    builder.create_engine(trt_model_path)
