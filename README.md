# collider prediction using acceleration sensor
[Collider detection using acceleration sensor](https://dacon.io/competitions/official/235614/overview/) task was to predict collider 2D position(x, y), mass and velocity.<br/>

## Summary of approach
- LB Score: **Entered top 7% (Public 0.0063, Private: 0.00662)**

## Data
- Data visualization<br/>
![acceleration sensor 1](./img/s1.png)<br/>
![acceleration sensor 2](./img/s2.png)<br/>
![acceleration sensor 3](./img/s3.png)<br/>
![acceleration sensor 4](./img/s4.png)<br/>

## Method
- Residual CNN for time-series (M)
- Multi-layer perceptron (X, Y, V)
- Model average ensemble
- Permutation feature selection

## Feature engineering
- Distance Feature using signal arrival time delta
- Original Signal<br/>
![fft](./img/signal.png)<br/>
- Fast Fourier Transform (FFT)<br/>
![fft](./img/fft.png)<br/>
- Power density estimation (PSD)<br/>
![psd](./img/psd.png)<br/>
- Auto correlation<br/>
![autocorr](./img/autocorr.png)<br/>
- Spectrogram<br/>
![spectrogram](./img/spectrogram.png)<br/>
- Mel spectrogram (except)<br/>
![mel](./img/mel.png)<br/>
- Statistical information
- Peak distance mean/std
