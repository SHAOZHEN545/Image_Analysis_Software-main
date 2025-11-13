# Image Analysis Software

## Overview
Image Analysis Software is a wxPython-based desktop application for processing absorption images from cold-atom experiments. The main `ImageUI` window provides a live view of raw images, atom number fits, and derived parameters while the program watches a directory for new data, performs automated fitting, and exports results for further analysis.【F:ImageUI.py†L271-L4566】 Supporting tools include multi-run fitting workflows, live trend plots, average-image previews, and a standalone ThorCam capture utility.【F:fitting_window.py†L1-L520】【F:ImageUI.py†L103-L220】【F:average_preview.py†L1-L200】【F:camera.py†L1-L17】

## Key Capabilities
- **Multi-format image ingestion** – load Andor `.aia`, multi-page `.tif`, or 3-layer FITS files containing probe-with-atom, probe-without-atom, and dark frames; each file is converted into an optical-density absorption image for analysis.【F:imgFunc_v7.py†L46-L135】
- **Automated directory monitoring** – watch experiment folders for new images, automatically refit them, and log the outputs while maintaining per-day subdirectories built from the configured base path.【F:ImageUI.py†L531-L688】【F:ImageUI.py†L4133-L4270】
- **Flexible fitting models** – choose Gaussian, fermionic, or bosonic 1D fits for single-shot analysis; the dedicated fitting window exposes additional models (linear, quadratic, exponential, MOT lifetime, damped harmonic, etc.) for single- and multi-run workflows, including temperature/PSD extraction and heat-map visualisation.【F:ImageUI.py†L568-L618】【F:fitting_window.py†L22-L520】【F:fit_functions.py†L7-L199】
- **Data export and trend tracking** – automatically append per-shot results to dated CSV files, generate live trend plots of widths/centers/atom number, and optionally open a large-font atom number display for quick monitoring.【F:ImageUI.py†L4321-L4448】【F:ImageUI.py†L103-L220】
- **Advanced preprocessing** – interactively adjust ROIs, enable normalization or OD limits, flip/rotate frames, apply median filtering, and preview the average of the most recent images with saturation corrections and inverse-variance weighting options.【F:ImageUI.py†L568-L760】【F:average_preview.py†L20-L200】
- **Camera acquisition** – launch the `camera.py` utility to control a ThorLabs camera (when the SDK is available) using the `thorcam_window` interface and Windows DLL setup helper.【F:camera.py†L1-L17】【F:windows_setup.py†L1-L45】

## Repository Layout
| Path | Description |
| ---- | ----------- |
| `ImageUI.py` | Main GUI application, directory watcher, fitting logic, exports, ROI tools, and auxiliary frames.【F:ImageUI.py†L271-L4566】 |
| `fitting_window.py` | Secondary window for single-run, multi-run, and heat-map fitting with configurable axes, units, and fit models.【F:fitting_window.py†L22-L520】 |
| `average_preview.py` | Average-image preview and processing workflow with saturation correction and weighting controls.【F:average_preview.py†L20-L200】 |
| `imgFunc_v7.py` | Low-level image readers, optical-density generation, and helper routines for normalization and statistics.【F:imgFunc_v7.py†L46-L142】 |
| `fit_functions.py` | Library of fitting functions and derived-value helpers used by the fitting window, including MOT lifetime analysis.【F:fit_functions.py†L7-L194】 |
| `Monitor.py` & `watchforchange.py` | Directory monitoring utilities built on `watchdog` to trigger automatic refits when new data arrives.【F:Monitor.py†L1-L97】 |
| `camera.py`, `thorcam_window.py`, `windows_setup.py` | Optional ThorCam capture UI and Windows PATH helper for the Thorlabs SDK.【F:camera.py†L1-L17】【F:windows_setup.py†L1-L45】 |
| `localPath.py` | Default base directory for new data folders (year/month/day/atom). Update this to match your environment.【F:localPath.py†L1-L3】【F:ImageUI.py†L531-L558】 |
| `requirements.txt` | Python dependencies required to run the software.【F:requirements.txt†L1-L11】 |

## Requirements
- Python 3.9 or newer (wxPython and modern SciPy stacks require a recent interpreter).
- Packages listed in `requirements.txt`, including scientific libraries (`numpy`, `scipy`, `matplotlib`, `astropy`, `scikit-image`, `scikit-learn`), GUI dependencies (`wxPython`), and the directory watcher (`watchdog`). Install `thorlabs_tsi_sdk` only if you intend to use the ThorCam capture tool.【F:requirements.txt†L1-L11】
- Windows is the primary target (default paths and DLL handling are Windows-centric), but the code relies on cross-platform Python libraries and can be adapted to other OSes with compatible camera drivers.

## Installation
1. **Clone the repository** and create a virtual environment (recommended):
   ```bash
   git clone <repo-url>
   cd Image_Analysis_Software
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Optional – ThorCam support**:
   - Download the Thorlabs ThorCam SDK from the vendor portal.
   - Copy the SDK’s `SDK` and `dlls` folders into the repository directory.
   - When launching the capture tool or any module that imports `thorlabs_tsi_sdk`, call `windows_setup.configure_path()` (the capture UI handles this for you) so the DLLs are temporarily added to `PATH`.【F:windows_setup.py†L1-L45】

## Configuration
- **Default data location** – edit `LOCAL_PATH` in `localPath.py` to point to the root of your experiment’s image archive. On startup, `ImageUI` creates a dated folder structure (`<base>/<year>/<month>/<day>/<atom>/`) and watches it for new images.【F:localPath.py†L1-L3】【F:ImageUI.py†L531-L558】
- **Instrument constants** – the GUI exposes fields for magnification, pixel size, detuning, and atom species. These values determine the pixel-to-distance conversion and photon-scattering cross section used for atom number calculations.【F:ImageUI.py†L291-L366】【F:ImageUI.py†L2610-L2636】
- **Expected file size** – adjust the “Auto Fitting” controls in the GUI if your camera output size differs; this guards against processing incomplete files.【F:ImageUI.py†L622-L688】

## Data Preparation
- Each acquired image file must contain three layers: probe with atoms, probe without atoms, and dark reference. The software converts these into optical-density images and applies optional normalization before fitting.【F:imgFunc_v7.py†L72-L135】【F:ImageUI.py†L568-L760】
- Supported formats include Andor `.aia`, multi-frame `.tif`, and FITS files; select the correct type using the **File Type** radio buttons before processing.【F:ImageUI.py†L520-L538】【F:imgFunc_v7.py†L46-L120】
- For multi-run analysis, prepare `Variable List.txt`, `Parameter List.txt`, and optional `Parameter 2` lists with one value per line; the fitting window can also read these values directly from text boxes or files.【F:fitting_window.py†L256-L399】

## Running the Application
Launch the analysis GUI with:
```bash
python ImageUI.py
```
The main window opens with several panels:

1. **Settings** – choose the file type, fit model (Gaussian/Fermion/Boson), enable normalization, radial averaging, or optical-density limits, and adjust ROI/magnification/atom species settings.【F:ImageUI.py†L520-L618】
2. **Auto Fitting** – select the image folder, choose the output “Data” directory for CSV exports, toggle monitoring with the **Start/Watching** button, and review the image list populated from the watched folder.【F:ImageUI.py†L622-L688】【F:ImageUI.py†L4133-L4170】
3. **Tools** – open the advanced fitting window, launch live trend plots, or average recent images before applying fits.【F:ImageUI.py†L692-L719】
4. **Image Display** – view the current absorption image alongside horizontal/vertical line-outs, switch between raw layers, flip/rotate images, and interactively resize the ROI rectangle by dragging on the canvas.【F:ImageUI.py†L725-L760】【F:ImageUI.py†L2649-L2706】

### Typical Workflow
1. Update `localPath.py` and select the correct image directory if necessary.
2. Set the desired atom species, magnification, and ROI; choose your fitting model and normalization options.
3. Press **Start** to begin watching the folder. New images trigger automatic loading, fitting, and display updates; results are appended to the daily CSV when auto-export is enabled.【F:ImageUI.py†L4133-L4270】【F:ImageUI.py†L4321-L4440】
4. Use the **Image List** to revisit earlier shots, manually adjust settings, and press **Save Fit** to re-export if needed.【F:ImageUI.py†L622-L688】【F:ImageUI.py†L657-L670】

## Automated Monitoring & Exports
The `Monitor` class relies on `watchdog` to detect new files, ensure they reach the expected size, and invoke `autoRun`, which refreshes the date-based directory if midnight passes and executes `fitImage` for the latest file.【F:Monitor.py†L12-L97】【F:ImageUI.py†L4203-L4270】 When auto-export is active, `snippetCommunicate` appends a rich set of fit parameters (atom number, widths, temperatures, degeneracy metrics, and multi-run variables) to a daily CSV in the configured Data folder.【F:ImageUI.py†L4321-L4448】

Live diagnostic tools include:
- **Trend plots** – plot atom number, widths, or centers over time with optional rolling averages, reset controls, and error bars where available.【F:ImageUI.py†L103-L220】
- **Atom number display** – show the fitted atom number in a resizable stand-alone window for quick lab monitoring.【F:ImageUI.py†L41-L100】

## Multi-Run Fitting Window
Select **Tools → Fitting** to open the dedicated fitting window. Key features include:
- **Single Run / Multi-Run / Heat-map modes** – process the current ROI once, iterate through a sequence of shots with associated variable and parameter lists, or visualise two-dimensional scans as heat maps.【F:fitting_window.py†L218-L454】
- **Configurable axes and units** – assign labels, unit families, and scaling factors for the scanned variables and fit parameters, with optional offsets and text/file inputs.【F:fitting_window.py†L256-L399】
- **Fit library** – choose from built-in models (linear, quadratic, Gaussian, Lorentzian, inverse, temperature & PSD, MOT lifetime, damped harmonic oscillator, etc.) and supply initial guesses per plot.【F:fitting_window.py†L22-L520】【F:fit_functions.py†L7-L199】
- **Derived metrics & exports** – store processed columns (atom number, centers, widths, degeneracy parameters) and save results to CSVs in the dated `Desktop/Data/...` directory configured on launch.【F:fitting_window.py†L116-L203】

## Average Preview & Preprocessing
The **Avg Images** tool opens a window for averaging the latest N shots before fitting. It supports per-shot optical-density averaging, pooled transmission calculations, saturation corrections (with customizable `I_sat`, Γ, and detuning), reference/ transmission guards, and inverse-variance weighting. Accepted averages replace the live image shown in the main GUI and update subsequent fits.【F:average_preview.py†L20-L200】

## ThorCam Capture Utility
Run `python camera.py` to launch the ThorCam acquisition window. The helper ensures `thorlabs_tsi_sdk` DLLs found in the local `SDK` and `dlls` folders are added to `PATH`, enabling camera control without system-wide installation.【F:camera.py†L1-L17】【F:windows_setup.py†L1-L45】

## Troubleshooting Tips
- If auto-run stops reacting, press **Start** again to restart the `watchdog` observer; the GUI disables path selection while watching to prevent accidental moves.【F:ImageUI.py†L4133-L4170】
- Ensure files contain all three exposure layers; otherwise `createAbsorbImg` raises an error indicating the missing planes.【F:imgFunc_v7.py†L72-L135】
- When migrating to a new PC, update `localPath.py` and confirm the Data export directory exists; the program creates missing folders automatically during startup and when saving results.【F:ImageUI.py†L531-L558】【F:ImageUI.py†L4321-L4347】

## Running the ThorCam Capture Tool
```bash
python camera.py
```
This opens the ThorCam window using wxPython. Install the Thorlabs SDK and ensure the DLL helper is configured as described above before launching the tool.【F:camera.py†L1-L17】【F:windows_setup.py†L1-L45】

## License
Include your project’s licensing information here if applicable.
