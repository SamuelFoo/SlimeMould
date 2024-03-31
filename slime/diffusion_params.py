class DiffusionParams:
    def __init__(
        self,
        diffusion_threshold=3.5,
        diffusion_decay_rate=1.26,
        distance_for_diffusion_threshold=55,
        moving_threshold=1,
        max_ph_threshold=5.5,
        max_ph_increase_step=0.2,
    ):
        # Diffusion Parameters
        self.diffusion_threshold = diffusion_threshold
        self.diffusion_decay_rate = diffusion_decay_rate
        self.distance_for_diffusion_threshold = distance_for_diffusion_threshold
        self.moving_threshold = moving_threshold
        self.max_ph_threshold = max_ph_threshold
        self.max_ph_increase_step = max_ph_increase_step


DEFAULT_DIFFUSION_PARAMS = DiffusionParams()
