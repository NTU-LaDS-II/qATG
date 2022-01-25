# quantum-ATG

## qatg
A qatg object:
```python=
generator = qatg(circuitSize: int, basisGateSet: list[qGate], \
			quantumRegisterName: str = 'q', classicalRegisterName: str = 'c', \
			gridSlice: int = 21, gradientDescentSearchTime: int = 800, gradientDescentStep: float = 0.01, \
			maxTestTemplateSize: int = 50, minRequiredEffectSize: float = 3)
```
```basisGateSet``` can be an universal gate set.
```python=
generator.configurationSimulationSetup(oneQubitErrorProb = 0.001, twoQubitErrorProb = 0.1, \
			zeroReadoutErrorProb = [0.985, 0.015], oneReadoutErrorProb = [0.015, 0.985], \
			targetAlpha: float = 0.99, targetBeta: float = 0.999, \
			simulationShots: int = 200000, testSampleTime: int = 10000)
```
to setup the simulation related parameters. Those parameters will type into ```qatgConfiguration``` that is produced by ```qatg```.

## qatgConfiguration
* To get the configuration:
```python=
configurationList = generator.getTestConfiguration(self, singleFaultList, twoFaultList, \
			singleInitialState: np.array = np.array([1, 0]), twoInitialState: np.array = np.array([1, 0, 0, 0]), simulateConfiguration: bool = True)
```
* The type of ```configurationList``` is ```list[qatgConfiguration]```.
* ```configuration.simulate()``` to simulate a ```qatgConfiguration```.

## Faults
* ```singleFaultList``` is a list of single-qubit faults that you want to put into qatg.
* ```twoFaultList``` is a list of two-qubit faults that you want to put into qatg.
* To generate a fault, inherit ```qatgFault```, and override ```getOriginalGateParameters(self)``` and ```getFaulty(self, parameters)```.
* Initialization of ```qatgFault```:
```python=
def __init__(self, gateType, qubit, description = None)
```
* ```gateType``` is the gate type that the fault focuses on, should check if ```issubclass(gateType, qiskit.circuit.gate.Gate)```.
* ```qubit``` is the qubit(s) that the fault will apply on. Can be a integer or a list of integer. Should match the number of qubits the gate acts on.
* ```description``` is the self-defined description for printing convenience.
* ```getOriginalGateParameters(self)``` should return a parameter list, which is from the original gate (or faultfree gate) of the fault. For example, for ```UGate``` you should return ```[theta, phi, lam]```, and for ```CXGate``` you should return ```[]```.
* ```getFaulty(self, parameters)``` should return your faulty gate if the faultfree gate's parameters are ```parameters```. Should check the return type fits ```issubclass(return, qiskit.circuit.gate.Gate)```.
* Note that since qiskit is stupid, it can't figure out a complex circuit's effective matrix if you didn't assign it. Since we used ```to_matrix()``` in our program, we would like you to implement ```faultyGate.__array___``` so we can call it by ```to_matrix()```. An example is in ```example.py```.