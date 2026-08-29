# Thesis Algorithm (Original Proposal)

This is the algorithm as proposed in the thesis document, section 3.2.1. It is
the formal specification the codebase implements — `utils/trading.py` and
`training/train.py::run_epoch` are Steps 4-8; `utils/pipeline.py` is Steps
1-2; `models/gru_model.py` is Step 3. Kept verbatim as the reference to check
code against, not to be edited to match code drift — if the two disagree,
that's a discrepancy to resolve deliberately, not silently.

---

## 3.2.1 Proposed Algorithm

**Step 0: START**

**Step 1: Historical Stock Data Input and Preprocessing**

*Given:* $P_t$ historical stock price at time $t$, $X$ historical stock data, $L$ window length

Collect historical stock market data consisting of Open, High, Low, Close, and Volume values. Missing values are handled, and the data is normalized to ensure consistent scaling. The time-series data is then transformed into sliding-window sequences, where previous observations are used as input for forecasting.

$$X_t = [X_{t-L}, X_{t-L+1}, \dots, X_{t-1}]$$

**Step 2: Market Return Computation**

*Given:* $P_t$ current stock price, $P_{t+1}$ next stock price

Compute the actual market return between two consecutive time steps:

$$return_t = \frac{P_{t+1} - P_t}{P_t}$$

This return value will be used as the basis for simulated trading profit.

**Step 3: GRU Forecasting Layer**

*Given:* $X_t$ input sequence, $h_{t-1}$ previous hidden state

Pass the input sequence into the GRU model to generate the predicted return:

$$\hat{r}_t = GRU(X_t)$$

The GRU processes the sequential input through its update and reset gates to capture temporal dependencies in the historical stock data.

### OBJECTIVE 1 – Profit-Aware Loss Learning

**Step 4: Direction-Sensitive Trading Signal Generation**

*Given:* $\hat{r}_t$ predicted return, $\alpha$ scaling factor

Convert the predicted return into a continuous trading signal using the hyperbolic tangent function:

$$signal_t = \tanh(\alpha \hat{r}_t)$$

A positive signal indicates a buy-oriented position, while a negative signal indicates a sell-oriented position. This allows the model output to be interpreted as a trading decision.

### OBJECTIVE 2 – Direction-Sensitive Trading Signal

**Step 5: Trading Friction Penalty Computation**

*Given:* $signal_t$ current trading signal, $signal_{t-1}$ previous trading signal, $c$ transaction cost rate

Compute the trading friction penalty based on the change in trading position:

$$cost_t = c\,|signal_t - signal_{t-1}|$$

This penalty represents practical trading costs such as transaction fees, bid-ask spread, and slippage. The penalty increases when the model frequently changes its trading position.

### OBJECTIVE 3 – Trading Friction

**Step 6: Cost-Adjusted Simulated Trading Profit Computation**

*Given:* $signal_t$ trading signal, $return_t$ actual market return, $cost_t$ trading friction penalty

Compute the simulated trading profit after deducting transaction cost:

$$profit_t = signal_t \times return_t - cost_t$$

This step allows the enhanced GRU to measure financial usefulness based on cost-adjusted simulated profit instead of prediction error alone.

### OBJECTIVE 3 – Trading Friction

**Step 7: Profit-Aware Loss Calculation**

*Given:* $profit_t$ cost-adjusted simulated trading profit, $N$ number of samples

The training loss is computed as the negative mean of cost-adjusted simulated trading profit:

$$Loss = -\frac{1}{N}\sum_{t=1}^{N} profit_t$$

By minimizing this loss, the enhanced GRU is trained to maximize simulated trading profit while accounting for trading friction.

### OBJECTIVE 1 – Profit-Aware Loss Learning

**Step 8: Backpropagation and Parameter Update**

*Given:* $Loss$ profit-aware loss

Compute the gradients of the loss with respect to the GRU model parameters and update the parameters using an optimizer.

$$\theta = \theta - \eta \nabla_\theta Loss$$

where $\theta$ represents the model parameters and $\eta$ represents the learning rate.

### OBJECTIVE 1 – Profit-Aware Loss Learning

**Step 9: Iteration**

Repeat Steps 3 to 8 for all batches and epochs until training is complete.

**Step 10: Output**

Return the trained enhanced GRU forecasting model.

**Step 11: END**
