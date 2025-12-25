import argparse
import configparser
from pathlib import Path

CORE_CFG_PATH = Path(__file__).parents[1] / "config.ini"


class PipelineCLI:
    # set the interpolation to ExtendedInterpolation to allow referencing variables in different sections
    SRC_CONFIG = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(), allow_no_value=True
    )
    SRC_CONFIG.read(CORE_CFG_PATH)
    # Map CLI argument names to (section, key, type_fn)
    CONFIG_MAPPABLES = {
        "resume": ("sampler_config", "resume_run"),
        "nlive": ("sampler_config", "Npoints"),
        "seed": ("sampler_config", "seed"),
    }

    def __init__(self):
        """
        Command line interface for the per-pulsar analysis pipeline.
        This class handles the command line arguments and validates them.
        """
        self.parser = argparse.ArgumentParser(
            description="Per-pulsar Analysis pipeline",
        )
        self._add_inicfg_args()
        self._add_common_args()
        self._add_output_args()
        self._add_mappable_args()

        self.args = self.parser.parse_args()
        self._validate_args(self.args)
        local_cfg_path = Path(self.args.out_directory) / f"{self.args.tag}_cfg.ini"
        # automatically read the local config file; invalid paths will be silently ignored
        self.load_config(local_cfg_path)
        if self.args.src_config:
            cfgs = [
                cfg for sub_cfg_list in self.args.src_config for cfg in sub_cfg_list
            ]
            # Load the configuration files (note that the last one takes precedence)
            self.load_config(cfgs)
        # Override the parameters defined in the CONFIG_MAPPABLES with CLI arguments
        self._merge_cli_overrides()
        # Snapshot the current configuration to a local file
        self.write_config(local_cfg_path)

    def _add_inicfg_args(self):
        inicfg = self.parser.add_argument_group("Configuration File")
        inicfg.add_argument(
            "--src_config",
            "-c",
            help="Path to the src config.ini file; Default=None, which leads to using the default config. If provided, it will override the conflicting keys in the default config. Multiple config files (e.g. -c config1.ini config2.ini or -c config1.ini -c config2.ini) can be provided, with the last one taking precedence (see the configparser documentation); this is useful for overriding existing config files with new ones for multiple objects.",
            default=[],
            nargs="*",
            action="append",
            type=str,
        )

    def _add_common_args(self):
        common = self.parser.add_argument_group("Data Specification")
        common.add_argument(
            "--parfile", type=str, required=True, help="Path to the parameter par file"
        )
        common.add_argument(
            "--timfile", type=str, required=True, help="Path to the TOAs tim file"
        )
        common.add_argument(
            "--nume_impl",
            default="numpy",
            choices=["mpmath", "numpy"],
            help="Numerical implementation to use",
        )
        common.add_argument(
            "--tempo_residuals",
            "-tres",
            default=False,
            action="store_true",
            help="use tempo phase residuals instead of manually calculated ones",
        )
        common.add_argument(
            "--run_glitch_psr",
            "-rgpsr",
            default=False,
            action="store_true",
            help="run the analysis on a glitching pulsar",
        )

    def _add_output_args(self):
        sampling = self.parser.add_argument_group("Output Configuration")
        sampling.add_argument(
            "--out_directory", help="output directory", default="./outdir/", type=str
        )
        sampling.add_argument(
            "--tag", help="tag to include in saving information", default="real"
        )

    def _validate_args(self, args):
        if not Path(args.parfile).exists():
            raise FileNotFoundError(f"Par file not found: {args.parfile}")
        if not Path(args.timfile).exists():
            raise FileNotFoundError(f"Tim file not found: {args.timfile}")

    @classmethod
    def load_config(cls, config_path):
        """
        Load the configuration from a src_config file.
        """
        cls.SRC_CONFIG.read(config_path)

    @classmethod
    def write_config(cls, output_path):
        """
        Write the current configuration to a file.
        """
        with open(output_path, "w") as configfile:
            cls.SRC_CONFIG.write(configfile)

    def _add_mappable_args(self):
        """Add arguments based on config mappables"""
        for arg_name, (section, key) in self.CONFIG_MAPPABLES.items():
            self.parser.add_argument(
                f"--{arg_name.replace('_', '-')}",
                help=f"Override {section}.{key} from config",
            )

    def _merge_cli_overrides(self):
        """
        Merge command line arguments with the configuration file.
        This allows command line arguments to override the config file settings.
        """

        for arg_name, (section, key) in self.CONFIG_MAPPABLES.items():
            cli_value = getattr(self.args, arg_name, None)
            if cli_value is not None:
                self.SRC_CONFIG.set(section, key, str(cli_value))
