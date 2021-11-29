# quantum-ATG

## Initialize
```python=
generator = ATPG(circuit_size, gate_set, alpha, beta, grid_slice, search_time, step, sample_time, max_element, min_required_effect_size)
```
* `circuit_size`: the number of the qubits of the QC
* `gate_set`: the gate set supported by the QC. Must be universal.
* `alpha`: 1-test escape. Default 0.99
* `beta`: 1-overkill. Default 0.999
* `grid_slice`: the slices of each parameters that the grid search uses. Default 11
* `search_time`: the search time of the fine-tune gradient descent. Default 800
* `step`: the step size of the fine-tune gradient descent. Default 0.01
* `sample_time`: the sample time while simulating the configuration. Default 10000
* `max_element`: the max test element number of a test template. Default 50
* `min_required_effect_size`: the minimum effect size required to complete a test template.
	* If the number of the test elements is not larger than `max_element`, but the effect size is over `min_required_effect_size`, the test template is completed.
	* If the number of the test elements exceeds the `max_element`, the test template is completed no matter how much the effect size is.