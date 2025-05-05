import torch
import torch.nn.functional as F
from holoscan.core import ExecutionContext, InputContext, Operator, OperatorSpec, OutputContext


class GsplatLoaderOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        # Initialize gaussian splat parameters
        self.n_points = 10000

        # Generate random positions between -1 and 1 as torch tensors on CUDA
        self.means = torch.rand(self.n_points, 3, device="cuda") * 10 - 5  # Scale to [-1, 1]

        # Initialize scales (size of gaussians) as torch tensors - small random values
        self.scales = torch.rand(self.n_points, 3, device="cuda") * 0.03  # Scale to [0.02, 0.05]

        # Initialize rotations as identity quaternions [w,x,y,z] as torch tensors
        self.quats = torch.zeros(self.n_points, 4, device="cuda")
        self.quats[:, 0] = 1.0  # w component = 1 for identity rotation

        # Initialize random colors with opacity as torch tensors
        self.colors = torch.rand(self.n_points, 3, device="cuda")
        self.opacities = torch.ones(self.n_points, 1, device="cuda")

        ckpt_paths = [
            "/workspace/holohub/applications/xr_gsplat/ckpts/ckpt_6999_rank0.pt",
            "/workspace/holohub/applications/xr_gsplat/ckpts/ckpt_29999_rank0.pt",
        ]
        if ckpt_paths is not None:
            self.means, self.quats, self.scales, self.opacities, self.sh0, self.shN = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for ckpt_path in ckpt_paths:
                ckpt = torch.load(ckpt_path, map_location="cuda")["splats"]
                self.means.append(ckpt["means"])
                self.quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
                self.scales.append(torch.exp(ckpt["scales"]))
                self.opacities.append(torch.sigmoid(ckpt["opacities"]))
                self.sh0.append(ckpt["sh0"])
                self.shN.append(ckpt["shN"])
            self.means = torch.cat(self.means, dim=0)
            self.quats = torch.cat(self.quats, dim=0)
            self.scales = torch.cat(self.scales, dim=0)
            print("scales shape:")
            print(self.scales.shape)
            self.opacities = torch.cat(self.opacities, dim=0)
            self.sh0 = torch.cat(self.sh0, dim=0)
            self.shN = torch.cat(self.shN, dim=0)
            self.colors = torch.cat([self.sh0, self.shN], dim=-2)

    def setup(self, spec: OperatorSpec):
        # Define optional input for view output from HolovizOp
        spec.output("splats")

    def start(self):
        pass

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        splats = dict()
        splats["means"] = self.means
        splats["quats"] = self.quats
        splats["scales"] = self.scales
        splats["opacities"] = self.opacities
        splats["colors"] = self.colors

        op_output.emit(splats, "splats")

    def stop(self):
        pass
