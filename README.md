# PIQE
Pulse calibration on the IBM quantum Experience

### Abstract
It was 1981, at the Physics of Computation Conference, when Richard Feynman talked about the necessity to create computers based on the principles of the quantum mechanics to better investigate atomic and subatomic phenomena. Computers able to take advantage of effects as entanglement, interference and superposition of states; effects unknown to the Informatics of the time. In this way the idea of quantum computer and quantum computing took place. Since then, physicists, mathematicians, engineers and computer scientists had to deal with harsh tasks; for example: the difficulties of building extremely sophisticated systems and to conceive algorithms which could be executed on real quantum device. Forty years later, one of the main promoters of that historical conference, IBM, offers a free cloud service to learn to use quantum computers, to interact with them and to test algorithms based on quantum computing concepts. The work presented here was possible thanks to this service.
The main topic of this thesis is the quantum bit, also known as qubit. In the first chapter, the concept of qubit is introduced as a mere mathematical object; such an abstraction was used by the pioneer of this new branch of Science. In particular, I would like to mention Peter Shor, David Deutsch and Richard Jozsa whose algorithms are some of the most famous ones. In fact, just from a theoretical point of view, there is the chance to argue about what is possible and what is impossible to do with a qubit. How to take advantage from a system which could be in any one of an infinite number of linear combinations of states is a topic of particular interest. It is convenient to remember here, that a “measure” in quantum mechanics is a projective operation; this is possibly the most bizarre axiom of this subject. In the same fashion of classical bits, we can obtain just two different results from applying a measurement to a qubit: 0 or 1, which are commonly indicated with |0> and |1> in the Dirac’s notation. After such a theoretical approach, a more practical topic is discussed: the DiVincenzo criteria; all of whom establish the features needed for a device to be considered a quantum computer. Since 2000, these criteria have never been changed and some companies, such as Google, IBM, Rigetti and D-wave, have been able to fulfill all the criteria’s requests. They used quantum dots, NV diamonds, NMR in molecules, ion traps, photons and superconducting circuits. About the latter of this list will be provided a complete explanation of the Hamiltonian which describe the system. The choice of this deep interest for the superconducting qubit has been suggested by the fact that they are the ones used by IBM and so is the backend employed in the experiments of this work. Superconducting quantum computers seem to be the most scalable system; indeed, IBM has recently announced that could be able to build a 1121 qubits computer in 2023; its name should be ibmq- condor. At the moment, their computer with the largest number of qubits is made of 65.
At the end of chapter 1, IBM Quantum Experience, Qiskit and QiskitPulse are introduced. These two frameworks provide the necessary Python libraries to write and execute algorithms on real quantum devices. In particular, Qiskit is useful for the construction of the quantum circuits on which algorithms can be tested; QiskitPulse, instead, allows the creation of pulse-level schedules; thanks to this approach, the user is enabled to have a greater control on the qubits. In chapter 2 it is showed how to use these features; it will be illustrated a characterization experiment: finding the resonance frequency of the circuit employing spectroscopy methods, and measuring characteristic times of relaxation and decoherece (T1 and T2).
In chapter 3 an optimization process will be described. It is here discussed how to speed up the operation applied to a qubit (quantum gates) and how to create high-fidelity pulse to implement such gates. This kind of task is essential to be able to run algorithms on a quantum computer.
In chapter 4 it is provided an example of quantum classifier. Roughly speaking, a program based on machine learning techniques, which is able to estimate the state of a qubit after an arbitrary sequence
on gate applications. It can be an example of the advantages offered by quantum computers, which are even more convenient when entanglement and interference are properly used.
