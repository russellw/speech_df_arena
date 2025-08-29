:: 1) Temporarily downgrade pip so it accepts old omegaconf metadata
python -m pip install "pip<24.1"
pip --version

:: 2) Ensure build tools libs are in place (you already installed some, but this is safe):
pip install "cython<3" ninja cmake

:: 3) Preinstall the exact deps fairseq wants (no isolation so it uses your cython/ninja)
pip install --no-build-isolation "omegaconf==2.0.6" "hydra-core==1.0.7"

:: 4) Now install fairseq 0.12.2 (still no isolation)
pip install --no-build-isolation "fairseq==0.12.2"

:: 5) Quick import test
python - <<EOF
import fairseq, omegaconf, hydra
print("fairseq:", getattr(fairseq, "__version__", "??"))
print("omegaconf:", omegaconf.__version__)
print("hydra-core:", hydra.__version__)
EOF

:: (Optional) Put pip back to latest once fairseq is in
python -m pip install -U pip
