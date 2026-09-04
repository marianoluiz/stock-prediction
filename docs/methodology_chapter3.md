# Chapter Three: Methodology

## 3.1 Research Design

This study adopts a quantitative experimental research design to evaluate the effectiveness of a profit-aware loss learning mechanism with a trading friction penalty as an enhancement to the Gated Recurrent Unit (GRU) model for stock market forecasting. The design is structured to systematically compare the performance of the proposed model against a baseline GRU configuration under controlled conditions.

Specifically, the study employs a comparative experimental framework, where model variants are trained and evaluated using identical datasets, feature-engineering procedures, and chronological data splits. The primary independent variable in this study is the training objective used by the model, while the dependent variables include both statistical prediction metrics and financial performance metrics.

The study uses historical daily stock market data retrieved programmatically rather than manually. For stocks covered by Yahoo Finance, data is retrieved using the `yfinance` Python library. For Philippine Stock Exchange (PSE)-listed companies not available through Yahoo Finance, historical Open, High, Low, Close, and Volume (OHLCV) data is instead retrieved from TradingView through the `tvDatafeed` client, allowing the study to include local equities alongside internationally listed ones under the same experimental pipeline. Retrieved data is cached locally as CSV files so that repeated experiment runs use an identical, reproducible dataset per symbol and date range.

Rather than feeding raw OHLCV values directly into the GRU, the raw Close price series is first transformed into a daily percentage return series, from which a compact set of engineered input features is derived: the same-day return, lagged returns at 1-, 5-, and 10-day offsets, a 20-day rolling volume z-score, and a 10-day rolling return volatility. This feature frame is then converted into fixed-length sliding-window sequences (default lookback of 30 trading days) that serve as GRU input, with the next-step return as the prediction target. Missing values introduced by percentage-change and rolling-window computations are dropped before sequence generation, and all feature values retain their natural numeric scale (returns, z-scores, and rolling statistics) rather than requiring a separate min-max or standard-score normalization pass, since the return-based representation is already scale-stable across assets.

The experimental setup consists of the following model configurations:

1. A **baseline GRU model** trained using Mean Squared Error (MSE) between predicted and actual returns.
2. An **enhanced GRU model** trained using a profit-aware loss with a trading friction penalty, combined with an MSE calibration term that keeps predicted returns near a realistic magnitude.

By isolating the training objective as the main enhancement component — while keeping the network architecture, input features, and data splits identical between configurations — the study ensures that any observed differences in performance can be attributed to the proposed profit-aware and friction-aware learning mechanism rather than unrelated architectural modifications.

The research design further incorporates a simulated trading evaluation framework, where model predictions are converted into continuous trading signals and assessed using financial performance indicators such as directional accuracy, cumulative return (both additive and compounding/geometric), and a Sharpe-like risk-adjusted return ratio. To reflect more realistic trading conditions, the evaluation also applies a transaction-cost-based trading friction penalty whenever the trading signal changes between consecutive time steps. This allows the study to evaluate not only predictive accuracy but also cost-adjusted financial viability.

To ensure robustness, the study applies a chronological train-validation-test split (70% / 15% / 15%) on time-series data, with no shuffling, to preserve temporal order and avoid data leakage. In addition, an expanding-window walk-forward validation scheme is available, which divides the trailing portion of each series into multiple contiguous evaluation folds and trains on an expanding history for each fold, allowing performance consistency to be checked across different market conditions within the dataset rather than relying on a single train/test cut.

Overall, the research design is intended to test whether aligning the GRU training objective with profit-based and cost-adjusted trading outcomes leads to better financial performance compared to traditional error-minimization approaches.

## 3.2 Proposed Algorithm and System Architecture

The study proposes an enhanced Gated Recurrent Unit (GRU) model that incorporates a profit-aware loss learning mechanism to align model optimization with financial trading performance. Unlike a conventional GRU model that minimizes statistical error using Mean Squared Error (MSE), the proposed approach optimizes a trading-based objective derived from simulated, cost-adjusted market returns.

### 3.2.1 Proposed Algorithm

**Step 0: START**

**Step 1: Historical Stock Data Input and Preprocessing**

*Given:* $P_t$ historical stock price at time $t$, $X$ historical stock data, $L$ window length

Collect historical daily OHLCV data (Yahoo Finance for internationally-listed equities, TradingView for PSE-listed equities). The Close price series is converted into a return series, engineered into a feature frame (return, lagged returns, rolling volume z-score, rolling volatility), and transformed into sliding-window sequences, where the previous $L$ observations are used as input for forecasting the next return.

$$X_t = [X_{t-L}, X_{t-L+1}, \dots, X_{t-1}]$$

**Step 2: Market Return Computation**

*Given:* $P_t$ current stock price, $P_{t+1}$ next stock price

Compute the actual market return between two consecutive time steps:

$$return_t = \frac{P_{t+1} - P_t}{P_t}$$

This return value is used both as the GRU's prediction target and as the basis for simulated trading profit.

**Step 3: GRU Forecasting Layer**

*Given:* $X_t$ input sequence, $h_{t-1}$ previous hidden state

Pass the input sequence into a 2-layer GRU (64 hidden units per layer, dropout regularization between layers) followed by layer normalization and a linear output head to generate the predicted return:

$$\hat{r}_t = GRU(X_t)$$

The GRU processes the sequential input through its update and reset gates to capture temporal dependencies in the historical stock data. To keep predictions on a realistic scale and prevent the downstream trading-signal function from saturating, the raw output is passed through a $\tanh$-scaled head, $\hat{r}_t = s \cdot \tanh(z_t)$, where $z_t$ is the raw linear output and $s$ is a scale set to a few standard deviations of the training return series.

### OBJECTIVE 1 – Profit-Aware Loss Learning

**Step 4: Direction-Sensitive Trading Signal Generation**

*Given:* $\hat{r}_t$ predicted return, $\alpha$ scaling factor

Convert the predicted return into a continuous trading signal using the hyperbolic tangent function:

$$signal_t = \tanh(\alpha \hat{r}_t)$$

A positive signal indicates a buy-oriented (long) position, a negative signal indicates a sell-oriented (short) position, and the magnitude of the signal represents position size / conviction, bounded within $(-1, 1)$. The scaling factor $\alpha$ is calibrated per asset as $\alpha = 1 / \mathrm{std}(return_{train})$, so that a one-standard-deviation predicted move maps to a signal of approximately $\pm 0.76$, keeping the signal sensitive to genuine directional moves regardless of an asset's volatility.

### OBJECTIVE 2 – Direction-Sensitive Trading Signal

**Step 5: Trading Friction Penalty Computation**

*Given:* $signal_t$ current trading signal, $signal_{t-1}$ previous trading signal, $c$ transaction cost rate

Compute the trading friction penalty based on the change in trading position:

$$cost_t = c\,|signal_t - signal_{t-1}|$$

This penalty represents practical trading costs such as transaction fees, bid-ask spread, and slippage. The penalty increases when the model frequently changes its trading position, including fractional position-size changes.

### OBJECTIVE 3 – Trading Friction

**Step 6: Cost-Adjusted Simulated Trading Profit Computation**

*Given:* $signal_t$ trading signal, $return_t$ actual market return, $cost_t$ trading friction penalty, $\hat{r}_t$ predicted return

Compute the simulated trading profit after deducting transaction cost:

$$profit_t = signal_t \times return_t - cost_t$$

$profit_t$ is the per-step quantity the loss in Step 7 aggregates: because it already folds in direction (via $signal_t \times return_t$) and cost (via $cost_t$), minimizing its negative mean trains the GRU against realized trading outcomes rather than against the raw distance between $\hat{r}_t$ and $return_t$.

In addition, define an MSE calibration term between the predicted and actual return:

$$MSE_t = (\hat{r}_t - return_t)^2$$

Because $\tanh$ saturates for large $|\alpha \hat{r}_t|$, a purely profit-driven objective has no incentive to keep $\hat{r}_t$ near the actual return's scale once the signal is already saturated at $\pm 1$ — the calibration term counteracts this by penalizing predictions that drift far from realistic return magnitudes, keeping gradients informative throughout training.

**Step 7: Profit-Aware Loss Calculation**

*Given:* $profit_t$ cost-adjusted simulated trading profit, $MSE_t$ MSE calibration term, $N$ number of samples, $\lambda$ calibration weight (default $\lambda = 0.7$)

The training loss is computed as the negative mean of cost-adjusted simulated trading profit, combined with the MSE calibration term:

$$Loss = -\frac{1}{N}\sum_{t=1}^{N} profit_t \;+\; \lambda \cdot \frac{1}{N}\sum_{t=1}^{N} MSE_t$$

By minimizing this loss, the enhanced GRU is trained to maximize simulated, cost-adjusted trading profit while the calibration term keeps predicted returns anchored to a realistic scale, preventing the trading signal from collapsing to a constant saturated value.

### OBJECTIVE 1 – Profit-Aware Loss Learning

**Step 8: Backpropagation and Parameter Update**

*Given:* $Loss$ profit-aware loss

Compute the gradients of the loss with respect to the GRU model parameters and update the parameters using the AdamW optimizer (with weight decay to further discourage prediction-magnitude divergence).

$$\theta = \theta - \eta \nabla_\theta Loss$$

where $\theta$ represents the model parameters and $\eta$ represents the learning rate.

**Step 9: Iteration**

Repeat Steps 3 to 8 for all batches and epochs until training is complete. Optionally, training stops early once a chosen validation metric (validation loss, cumulative profit, or directional accuracy) fails to improve for a set number of epochs, restoring the best-performing epoch's weights.

**Step 10: Output**

Return the trained enhanced GRU forecasting model.

**Step 11: END**

For the baseline configuration, Steps 4 through 7 are replaced by a direct Mean Squared Error loss between $\hat{r}_t$ and $return_t$, with all other steps (data preparation, GRU architecture, backpropagation, evaluation) held identical, so that the comparison isolates the training objective as the sole independent variable.

---

**Figure 3.1**
Proposed Algorithm Flowchart

### 3.2.2 System Architecture

**Figure 3.2**
System Architecture Input-Process-Output Diagram

The system architecture follows an input-process-output structure for the enhanced GRU forecasting model.

**Input.** Historical daily OHLCV data for the selected stock — retrieved via `yfinance` for internationally-listed equities, or via TradingView's `tvDatafeed` client for PSE-listed equities not covered by Yahoo Finance — cached locally as CSV for reproducibility.

**Process.** The process consists of: (1) data loading and cleaning; (2) return computation and feature engineering (lagged returns, rolling volume z-score, rolling volatility); (3) sliding-window sequence generation; (4) chronological train/validation/test splitting (70/15/15), or expanding-window walk-forward fold generation; (5) GRU forecasting (2-layer GRU, 64 hidden units, LayerNorm + linear head, optional $\tanh$-bounded output); (6) direction-sensitive trading signal generation; (7) trading friction penalty computation; (8) cost-adjusted simulated trading profit computation; (9) profit-aware loss calculation (with MSE calibration term); (10) backpropagation and parameter update via AdamW; and (11) evaluation on the held-out validation/test set.

**Output.** The output includes the trained enhanced GRU model (saved checkpoint), predicted returns, trading signals, cost-adjusted simulated trading profit, loss and profit curve plots, and comparative performance results — directional accuracy, additive and geometric cumulative return, a Sharpe-like ratio, and MSE/MAE/RMSE — evaluated side-by-side against the MSE-trained baseline model.

## 3.3 System Requirements

The study is designed to run on a standard personal computing setup capable of handling time-series processing and moderate deep learning workloads. The minimum and recommended hardware and software specifications are as follows:

**Table 3.1**
Hardware Requirement

| Hardware | Minimum Requirement | Recommended Requirement | Uses |
|---|---|---|---|
| Processor | Intel Core i5 8th Generation or equivalent | Intel Core i7 10th Generation or AMD Ryzen 5/7 or higher | Handles data preprocessing, sequence generation, model training, and experiment execution. |
| Memory (RAM) | 8 GB | 16 GB or higher | Stores datasets, training batches, model variables, and intermediate outputs during training. |
| Storage | 10 GB available disk space | SSD with at least 20 GB available disk space | Stores historical stock datasets, notebooks, trained model files, saved weights, and experiment results. |
| Graphics Processing Unit (GPU) | Integrated GPU or CPU-based training | NVIDIA GTX 1650, RTX 3050, or higher (CUDA-enabled) | Accelerates deep learning model training and testing when available. |
| Internet Connection | Required | Stable broadband connection | Used for retrieving historical stock market data from Yahoo Finance and TradingView. |

Table 3.1 presents the minimum and recommended hardware requirements for conducting the study. These specifications are needed to support stock data preprocessing, time-series sequence generation, GRU model training, experiment execution, and storage of datasets and model outputs. Since the study involves moderate deep learning workloads, a higher processor, larger memory capacity, and GPU support are recommended to improve training efficiency.

**Table 3.2**
Software Requirements

| Software / Library | Version | Uses |
|---|---|---|
| Python | 3.12 | Main programming language for model development. |
| PyTorch | 2.12.0 | Used to build, train, and evaluate the GRU model (with CUDA acceleration where available). |
| Pandas | 3.0.3 | Used for loading, cleaning, and preprocessing stock market data. |
| NumPy | 2.4.6 | Supports numerical computation, array operations, and evaluation-metric implementations. |
| Matplotlib | 3.10.9 | Used for loss-curve and profit-curve visualization. |
| yfinance | 1.3.0 | Used for retrieving historical stock data for Yahoo Finance-covered equities. |
| tvDatafeed (tvdatafeed-enhanced) | latest | Used for retrieving historical OHLCV data for PSE-listed equities via TradingView. |
| Jupyter Notebook / Google Colab | Latest version | Used for experiment development and testing. |
| Visual Studio Code | Latest version | Used as the main code editor. |
| Git / GitHub | Latest version | Used for version control and project collaboration. |

Table 3.2 presents the software requirements and libraries needed for the development, training, testing, and evaluation of the enhanced GRU model. Python serves as the main programming language, while PyTorch, Pandas, NumPy, and Matplotlib handle model implementation, data handling, preprocessing, and visualization; `yfinance` and `tvDatafeed` together provide historical data coverage across both internationally-listed and PSE-listed equities.

## 3.4 Theories, Methods, and Tools

### 3.4.1 Theories

**1. Profit-Aware Loss Learning**

Profit-aware loss learning is a training approach that optimizes a forecasting model based on financial usefulness rather than statistical prediction error alone. In conventional stock forecasting, models are commonly trained using loss functions such as Mean Squared Error (MSE) and Mean Absolute Error (MAE), which minimize the numerical difference between predicted and actual values. However, minimizing prediction error does not always result in profitable trading decisions, especially when the predicted price movement leads to an incorrect buy or sell position.

In this study, profit-aware loss learning is applied to the enhanced GRU model by using simulated, cost-adjusted trading profit as the primary basis of the training objective, combined with an MSE-based calibration term that keeps predicted returns near a realistic scale. Instead of only reducing the difference between predicted and actual values, the proposed model is trained to maximize simulated trading profit while remaining numerically well-calibrated. This direction is supported by studies that explored profit-guided, return-weighted, and reward-based loss functions, which emphasize that forecasting models should be evaluated not only by closeness to actual values but also by their usefulness in generating better trading outcomes.

**2. Direction-Sensitive Trading Signal**

A direction-sensitive trading signal is used to convert the predicted return of the GRU into a trading position. In stock market forecasting, the direction of the prediction is important because trading decisions depend on whether the market is expected to move upward or downward. Traditional symmetric loss functions, such as MSE and MAE, penalize prediction errors based on magnitude and may treat positive and negative deviations equally, even though their trading effects may differ.

In this study, the predicted return is transformed into a continuous trading signal using the hyperbolic tangent function, scaled by a per-asset factor $\alpha$ calibrated from the training return series' volatility. This signal represents the strength and direction of the model's trading decision as a continuous position size in $(-1, 1)$: values near $+1$ represent strong buy-oriented positions, values near $-1$ represent strong sell-oriented positions, and values near $0$ represent a near-flat position. By multiplying the trading signal with the actual market return, the model receives positive simulated profit when the predicted direction is correct and negative simulated profit when the predicted direction is wrong. This supports the study's goal of making the enhanced GRU more sensitive to directional mispredictions. Rajpal et al. (2025) supports this limitation by noting that symmetric loss functions may reduce sensitivity to directional shifts that are important in trading strategies.

**3. Trading Friction Penalty**

Trading friction refers to real-world costs and constraints that reduce actual trading profit. These include transaction fees, broker charges, bid-ask spread, market friction, and slippage. Slippage occurs when the expected price of a trade differs from the actual execution price. In practical trading environments, frequent buying and selling may reduce profitability because each change in position can introduce additional cost.

In this study, trading friction is represented through a differentiable transaction cost penalty proportional to the change in the (continuous) trading signal between consecutive time steps. This discourages unstable switching between buy and sell positions and makes the simulated trading outcome closer to practical trading conditions. This concept directly supports the study's third problem, which states that the baseline GRU does not incorporate trading friction and realistic profit maximization in its objective function.

**4. Cost-Adjusted Simulated Trading Profit**

Cost-adjusted simulated trading profit refers to the computed trading outcome after subtracting the trading friction penalty from the profit generated by the model's trading signal. A simple simulated profit may be computed by multiplying the model's trading signal with the actual market return. However, this may overestimate profitability because it does not consider the cost of changing positions.

In this study, cost-adjusted simulated trading profit is used as the basis of the proposed profit-aware loss function, and is also reported at evaluation time under two aggregation conventions: an **additive** cumulative return (each trade sized against the original capital) and a **geometric/compounding** cumulative return (each trade sized against the current balance, so losses shrink and gains expand subsequent trade sizes). By using cost-adjusted profit, the enhanced GRU is encouraged not only to predict profitable directions but also to reduce unnecessary position changes, optimizing for more realistic financial performance rather than theoretical profit alone.

### 3.4.2 Methods

**Table 3.3**
Methods and their Formulas

| Method | Formula |
|---|---|
| Market Return | $return_t = \dfrac{P_{t+1} - P_t}{P_t}$ |
| Direction-Sensitive Trading Signal | $signal_t = \tanh(\alpha \hat{r}_t)$, with $\alpha = 1/\mathrm{std}(return_{train})$ |
| Trading Friction Penalty | $cost_t = c\,\lvert signal_t - signal_{t-1}\rvert$ |
| Cost-Adjusted Trading Profit | $profit_t = signal_t \times return_t - cost_t$ |
| Profit-Aware Loss Learning | $Loss = -\dfrac{1}{N}\displaystyle\sum_{t=1}^{N} profit_t + \lambda \cdot \dfrac{1}{N}\displaystyle\sum_{t=1}^{N} (\hat{r}_t - return_t)^2$, $\ \lambda = 0.7$ |

**Method 1: Profit-Aware Loss Learning**

*SOP 1:* The existing GRU is optimized for statistical errors rather than actual financial gains.

*Objective 1:* To improve the financial viability of the GRU forecasting algorithm by shifting the training objective from purely statistical error minimization to a profit-oriented evaluation that incorporates simulated trading returns and financial performance metrics.

*Solution 1:* Profit-aware loss learning is integrated into the enhanced GRU to address the limitation of conventional statistical loss functions, such as Mean Squared Error (MSE) and Mean Absolute Error (MAE), which prioritize numerical prediction accuracy rather than financial profitability. In the baseline GRU, the model minimizes the difference between the predicted value and the actual value. However, this does not guarantee that the prediction will lead to a profitable trading decision. This limitation is aligned with Kar et al. (2025), who explored profit-guided loss functions for stock trading models by considering possible trading profit or loss rather than prediction error alone. Similarly, Guo and Qiu (2025) introduced a return-weighted loss function that gives greater importance to financially significant price movements.

In this study, the enhanced GRU replaces the MSE-based optimization objective with a profit-aware loss function. The model first generates predicted returns, converts them into trading signals, and computes cost-adjusted simulated trading profit. Because a purely profit-driven objective can allow the $\tanh$-based signal to saturate — at which point the gradient with respect to the raw prediction vanishes and the model can freeze on a single constant prediction — an MSE calibration term is added to the loss, weighted by $\lambda = 0.7$:

$$Loss = -\frac{1}{N}\sum_{t=1}^{N} profit_t + \lambda \cdot \frac{1}{N}\sum_{t=1}^{N}(\hat r_t - return_t)^2$$

By minimizing this loss, the enhanced GRU is encouraged to maximize simulated, cost-adjusted trading profit while the calibration term keeps its predictions numerically anchored to realistic return magnitudes. This allows the model to prioritize financially useful predictions while remaining trainable, and directly addresses the limitation that the existing GRU is optimized for statistical closeness instead of actual financial gain.

**Method 2: Direction-Sensitive Trading Signal Optimization**

*SOP 2:* The standard objective functions are symmetric and do not adequately penalize directional mispredictions.

*Objective 2:* To develop a custom profit-aware loss function that addresses the symmetric limitations of traditional objective functions by introducing directional penalties that strongly penalize incorrect trend predictions and reward predictions that correctly capture market movement.

*Solution 2:* Direction-sensitive trading signal optimization is applied to address the inability of symmetric objective functions to properly penalize wrong market direction predictions. Traditional losses such as MSE and MAE treat prediction errors based on numerical magnitude. This means that an overprediction and an underprediction with the same error size can receive the same penalty, even though their trading consequences may be different. Rajpal et al. (2025) emphasized that symmetric loss functions such as MSE can reduce sensitivity to sharp price movements and directional shifts that are important in trading strategies.

To address this limitation, the enhanced GRU converts the predicted return into a continuous trading signal using the hyperbolic tangent function:

$$signal_t = \tanh(\alpha \hat{r}_t)$$

where $\hat{r}_t$ represents the predicted return of the GRU and $\alpha$ is a scaling factor that controls the sensitivity of the trading signal, calibrated per asset as $\alpha = 1/\mathrm{std}(return_{train})$ so a one-standard-deviation predicted move maps to a signal magnitude of approximately $0.76$. The actual market return is computed as:

$$return_t = \frac{P_{t+1}-P_t}{P_t}$$

where $P_t$ represents the stock price at time $t$, and $P_{t+1}$ represents the stock price at the next time step.

The trading signal is then multiplied by the actual market return during simulated profit computation. If the predicted direction aligns with the actual market movement, the resulting profit becomes positive. If the predicted direction is incorrect, the resulting profit becomes negative. Through this mechanism, the enhanced GRU becomes direction-sensitive because incorrect trend predictions directly reduce simulated profit and increase the loss value.

**Method 3: Trading Friction Penalty**

*SOP 3:* The baseline GRU does not incorporate trading friction and realistic profit maximization in its objective function.

*Objective 3:* To incorporate realistic trading evaluation into the GRU training process by simulating trading outcomes under practical trading constraints, including transaction costs, slippage, trading signals, and profit-based evaluation.

*Solution 3:* A trading friction penalty is integrated into the simulated profit computation to represent practical trading costs. Trading friction refers to real-world costs and limitations that reduce trading profitability, such as transaction fees, broker charges, bid-ask spread, and slippage. Slippage occurs when the expected trade price differs from the actual execution price. These factors can significantly affect profitability, especially when a model frequently switches between buy and sell positions.

In the Philippine stock trading context, trading friction may include broker commission, value-added tax (VAT), PSE transaction fee, Securities Clearing Corporation of the Philippines (SCCP) fee, and sales tax for selling transactions. For example, COL Financial lists standard stock trading charges such as a 0.25% commission based on gross trade amount, 12% VAT on commission, 0.005% PSE transaction fee, 0.01% SCCP fee, and 0.1% sales tax for selling transactions (COL Financial, 2025). These fees show that trading profit may be reduced every time a position is opened, closed, or changed. Therefore, this study represents these practical costs through a simplified transaction cost rate $c$ (default $0.001$, i.e., 0.1% per unit of signal change) in the trading friction penalty.

To address this limitation, the enhanced GRU applies a transaction cost penalty whenever the trading signal changes between consecutive time steps. The penalty is computed as:

$$cost_t = c\,\lvert signal_t - signal_{t-1}\rvert$$

where $c$ represents the transaction cost rate, $signal_t$ represents the current trading signal, and $signal_{t-1}$ represents the previous trading signal. The cost-adjusted trading profit is then computed as:

$$profit_t = signal_t \times return_t - cost_t$$

where $profit_t$ represents the cost-adjusted simulated trading profit at time $t$, $signal_t$ represents the trading signal, $return_t$ represents the actual market return, and $cost_t$ represents the trading friction penalty.

This allows the proposed model to account for trading friction during both training and evaluation. If the model frequently changes its trading position, the cost penalty increases and reduces the simulated profit. As a result, the enhanced GRU is encouraged to produce more stable trading signals and optimize for realistic profit maximization instead of theoretical profit alone. This directly addresses the baseline GRU's lack of trading friction awareness in its objective function.

### 3.4.3 Tools

**Table 3.4**
Tools and their Uses in Research

| Tool | Uses |
|---|---|
| Visual Studio Code | Used for writing and managing the source code. |
| Jupyter Notebook | Used for testing formulas, preprocessing, and model experiments. |
| Google Colab | Used as an optional environment for GPU-assisted training. |
| Yahoo Finance / yfinance | Used to retrieve historical stock market data for internationally-listed equities. |
| TradingView / tvDatafeed | Used to retrieve historical OHLCV data for PSE-listed equities not covered by Yahoo Finance. |
| GitHub | Used for version control and project collaboration. |
| Matplotlib | Used to generate visualizations such as loss curves and trading (profit) performance graphs. |
