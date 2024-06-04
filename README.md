# qATG v0.6.1

## Introduction

This is an quantum automatic test generator, compatible with __**Qiskit version 0.39**__. The generator generates a configuration for each user-defined fault.
Configurations are generated by test templates, and templates are combined by test elements. One test element consists of one original gate and one activation gate. The use of activation gate is to enlarge the output probability difference of the original gate and the faulty gate.
Since the activation gate is solved assuming as it is an UGate, qATG requires the basis gate set of the quantum circuit should be 'universal'. That is, the quantum circuit can implement any UGate by using gates in the basis gate set.
Note that qATG assumes different faults for different qubits, and one consistent fault for one qubit.

## Installation

Installing from pip server, execute:

```bash
pip install qatg==0.6
```

Or, installing from local Github repository, at the root of cloned directory, execute:

```bash
pip install -e .
```

And then pip will handle the rest.

## Examples

Please refer to the examples in the examples folder.

* Example 1: Faulty UGate w/ self-defined bias fault. The basis gate set is {UGate}.
* Example 2: Faulty RXGate and RZGate w/ self-defined bias fault. The basis gate set is {RXGate, RZGate}.
* Example 3: Faulty CXGate w/ self-defined fault. The basis gate set is {UGate}. Note that currently we only care about those single-qubit gates in the basis gate set.
* Playground: Jupyter-lab playground with the above stuff.

## Detailed Class Parameters

### QATG

A `QATG` object is the main generator class. Below are the parameters to initialize the class object.

<details>
    <summary>Necessary Parameters</summary>
    <ul>
          <li><var>circuitSize</var>: the size of the qiskit circuit. Note that since qiskit starts their circuit from #0, if you want to construct a fault with qubits not starting from #0, please choose your circuit size wisely. For example, if you want a single-qubit fault on #3, since qubit #3 should exists the circuit size should at least be 4. This should be a positive integer.</li>
        <li><var>basisGateSet</var>: the basis gate set of the circuit. Should be "universal", that is, the circuit can implement any effective U gate by using the gates in the basis gate set. The <code>qatg</code> generator will transpile the optimal activation gate (which is a U gate) by <code>qiskit.transpile()</code>. This should be a list of <code>qiskit.circuit.library</code> gates.</li>
        <li><var>circuitInitializedStates</var>: this is a dictionary that records the initialized of the circuit with different qubit length. The key should be positive integers indicating the number of qubit the state concerns, and the value should be a normalized complex vector with length <code>2**key</code>, which states the initial state of the circuit for the concerned length of qubits. For example, this can be something like <code>{1: [1, 0], 2: [1, 0, 0, 0]}</code>, where in this case, the circuit is initialized to |0> for single-qubit gates, and |00> for two-qubit gates. Note that the order of qiskit qubits is quite different from some physics textbooks.</li>
    </ul>
</details>

<details>
    <summary>Qiskit Circuit Parameters</summary>
    <ul>
        <li><var>quantumRegisterName</var>: the quantum register name of the qiskit circuit. Is the parameter of <code>qiskit.QuantumRegister()</code>. The default value is 'q'.</li>
        <li><var>classicalRegisterName</var>: the classical register name of the qiskit circuit. Is the parameter of <code>qiskit.ClassicalRegister()</code>. The default value is 'c'.</li>
    </ul>
</details>

<details>
    <summary>Grid Search / Gradient Descent Parameters</summary>
    <ul>
        <li><var>gridSlice</var>: the slices of the grid while doing grid search, searching for parameters for sub-optimal activation gates. Currently the generator searches every U gate parameters in <code>numpy.linspace(-np.pi, np.pi, num=gridSlice, endpoint=True)</code>. This should be a positive integer and the default value is 11.</li>
        <li><var>gradientDescentMaxIteration</var>: the max iteration of gradient descent after the grid search. The generator does a bit gradient descent after the grid search for better fine tuning. This should be a positive integer and the default value is 1000.</li>
        <li><var>gradientDescentStep</var>: the step of each gradient descent. The gradient descent is performed by <code>x(t+1) = x(t) + s * gradient(score(x(t)))</code>, and the <var>s</var> is the step. The default value is 0.2.</li>
        <li><var>gradientMeasureStep</var>: since we cannot obtain the true gradient of the score function, we measure it by a discrete method <code>gradient(score(x(t))) = (x(t+e)-x(t))/e</code>. <var>e</var> is the measure step. The default value is 0.0001.</li>
        <li><var>gradientDeltaThreshold</var>: the ending criteria of gradient descent is that the 2-norm of the estimated gradient is less than this threshold. The default value is 1e-8.</li>
    </ul>
</details>

<details>
    <summary>Template Size Parameters</summary>
    <ul>
        <li><var>maxTestTemplateSize</var>: the maximum number of test elements in the test template. The default value is 50.</li>
        <li><var>minRequiredEffectSize</var>: the minimum required effect size for the generator to terminate. For smaller effect size, you might get a short test template, but a large repetition; for larger effect size, you might get a small repetition since the output probability difference is quite large for the faultfree and faulty circuit, but it requires long test template. The default value is 3.</li>
    </ul>
</details>

<details>
    <summary>Simulation Parameters</summary>
    <ul>
        <li><var>oneQubitErrorProb</var>: the depolarizing error of single-qubit gates while generating noise model during simulation. The default value is 0.001.</li>
        <li><var>twoQubitErrorProb</var>: the depolarizing error of two-qubit gates while generating noise model during simulation. The default value is 0.1.</li>
        <li><var>zeroReadoutErrorProb</var>: the readout error, called <code>qiskit.providers.aer.noise.errors.ReadoutError([self.zeroReadoutErrorProb, self.oneReadoutErrorProb])</code> while generating noise model during simulation. The default value is [0.985, 0.015].</li>
        <li><var>oneReadoutErrorProb</var> the readout error, called <code>qiskit.providers.aer.noise.errors.ReadoutError([self.zeroReadoutErrorProb, self.oneReadoutErrorProb])</code> while generating noise model during simulation. The default value is [0.015, 0.985].</li>
        <li><var>targetAlpha</var>: target 1-overkill of the test configuration. The default value is 0.99.</li>
        <li><var>targetBeta</var>: target 1-(test escape) of the test configuration. The default value is 0.999.</li>
        <li><var>simulationShots</var>: simulation shots while evaluating the faulty/faultfree distribution of the circuit. The default value is 200000.</li>
        <li><var>testSampleTime</var>: simulated overkill and simulated test escape will be evalutated using <var>testSampleTime</var> times of simulation. The default value is 10000.</li>
    </ul>
</details>

<details>
    <summary>Other Parameters</summary>
    <ul>
        <li><var>verbose</var>: whether additional information is printed during test configuration generation. The default value is False.</li>
    </ul>
</details>

To properly use the generator, initialized the generator as:

```python
from qatg import QATG

generator = QATG(arguments)
configurationList = generator.createTestConfiguration(FaultList)
```

Thus the generator will return a configuration list based on the fault list user provided. The fault list is a python list of user-defined fault objects. About how to properly define a fault, please refer to section `QATGFault`. The configurations in the configuration list is described using the class `QATGConfiguration`.

### QATGConfiguration

`QATGConfiguration` is to describe configurations that the `QATG` generator produces. `generator.createTestConfiguration` returns a list of configuration.
For a `configuration` in the configuration list, `print(configuration)` will print the detailed information of the configuration including target fault, circuit length, required repetition, total cost, the chi-value boundary, final effect size, simulated overkill and simulated test escape.
Also `configuration.circuit` will return the qiskit circuit of the configuration. By using `configuration.circuit.draw()` the program will draw the circuit on the terminal, which is a qiskit function. `draw('mpl')` will draw the circuit using matplotlib window.
Note that drawing circuit by `draw('mpl')` requires module `pylatexenc` and `ipympl` on jupyter-lab, and by installing `qatg` by `pip` will install that for you.

### QATGFault

`QATGFault` is the most important part of `QATG` since it describes the fault in a general way. A valid fault defined by users should inherit the class `QATGFault`. Below is an example of faulty U Gate, using bias fault as its faulty behavior.

```python
from qatg import QATGFault

class myUFault(QATGFault):
    def __init__(self, params):
        super(myUFault, self).__init__(qGate.UGate, 0, f"gateType: U, qubits: 0, params: {params}")
        self.params = params
    def createOriginalGate(self):
        return qGate.UGate(*self.params)
    def createFaultyGate(self, faultfreeGate):
        return qGate.UGate(faultfreeGate.params[0] - 0.1*np.pi, faultfreeGate.params[1], faultfreeGate.params[2]) # bias fault on theta
```

At least three functions are required in the user-defined fault class.

* `__init__(self, *args)`
    - This is the initializer of the class. By initializing the parent class `QATGFault` please use `super(<user-defined-class-name>, self).__init__(gateType, qubits, description = None)`.
    - `gateType` is the gate type of the fault. Note that since `qatg` is written based on `qiskit`, we currently only support gate types in `qiskit.circuit.library`. For example, `qiskit.circuit.library.UGate` for U gate, and `qiskit.circuit.library.CXGate` for CNOT gate. Please don't use `qiskit.circuit.gate.Gate` as your gate type, since every gate is a `Gate`, thus `qatg` simulator will treat every gate in your circuit faulty while doing faulty simulation, which might cause unexpected behavior if your `createFaultyGate` doesn't block anything you don't want.
    - `qubits` is the qubit list of the fault. For example, for a faulty CNOT gate, if the CNOT the user is discussing has its control bit on #0, and its target bit on #1, the `qubits` list would be `[0, 1]`. For single-qubit gates, either `[0]` or `0` is fine.
        + Note that `qatg` requires the length of `qubits` is the same as the qubits that the fault/gate type works on. Since `qatg` assumes different fault on different qubits, PLEASE declare different class objects for different qubits, despite the fact that they might have the same faulty behavior.
        + For example, if I want to test the U Gate fault on qubit #0 and #1, and they have the same faulty behavior. This is fine:

            ```python
            # fine
            class myUFault0(QATGFault):
                def __init__(self):
                    super(myUFault0, self).__init__(qGate.UGate, 0)
            class myUFault1(QATGFault):
                def __init__(self):
                    super(myUFault1, self).__init__(qGate.UGate, 1)

            configurationList = generator.createTestConfiguration([myUFault0(), myUFault1()])
            ```

            This is also fine:

            ```python
            # fine
            class myUFault(QATGFault):
                def __init__(self, qubit):
                    super(myUFault, self).__init__(qGate.UGate, qubit)

            configurationList = generator.createTestConfiguration([myUFault(0), myUFault(1)])
            ```

            But this is NOT WORKING:

            ```python
            # NOT WORKING
            class myUFault(QATGFault):
                def __init__(self):
                    super(myUFault, self).__init__(qGate.UGate, [0, 1])

            configurationList = generator.createTestConfiguration([myUFault()])
            ```

    - `description` is just for user convenience. The `description` will be printed while printing the configuration of the fault. It is optional and doesn't affect any computation in `qatg`. The default value is `None`.
* `createOriginalGate(self)`
    - User-defined fault class should inherit this function, with only `self` as parameter. Since there is a original gate in every test element of the configuration generated by `qatg`, this function requires user to return a faultfree gate with type `gateType`.
* `createFaultyGate(self, faultfreeGate)`
    - User-defined fault class should inherit this function, as it defines the faulty behavior of the fault class. The parameter of this function would be `self` and a `faultfreeGate`, while `faultfreeGate` will be a gate with type `gateType`.
    - The `faultfreeGate` would be any possible `gateType` since the function is called in `qatg`, with a `faultfreeGate` that `qatg` generates.
    - While those gates are qiskit gates, you can use `faultfreeGate.params` to obtain the parameters in the gate (see example #1, #2), or obtain the matrix by `faultfreeGate.to_matrix()`, change the matrix, and return a decent gate by using `qiskit.extensions.UnitaryGate(matrix)` (see example #3).

## Not-yet Stuff

* Ratio faults might be troublesome since `qatg` might generate some parameters such as `3*np.pi` while "transpiling". Since `3*np.pi` is actually `np.pi`, user might expect `np.pi * 0.9` instead of `3*np.pi * 0.9`. This is treated as a bug we currently are trying to fix.
* Numba accelerates is something we are working on.
* Currently `qatg` only supports single-qubit gates and CNOT gate. Adding more supports is something we are working on.

## Contact

Please contact NTU-LaDS-II (<lads427@gmail.com>) if you have any suggest or problems.
