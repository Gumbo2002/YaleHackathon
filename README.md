# YaleHackathon

DoraHacks Submission for Classification of QIRNG Sources or differentiation of quantum computers based upon patterns in randomly generated data.

## Table of Contents
- [Approach](#approach)
- [Results](#results)
- [Analysis](#analysis)
- [Conclusions](#conclusions)
- [Usage](#usage)

## Approach

Using classical techniques, analyzed data generated through arbitrarily noisy quantum simulator to determine potential routes for creating minimization functions in training neural networks. 

## Results

Two features were examined: number of maximally entangled qubits within a quantum system and the error probability on quantum channels (related to entanglement fidelity).

## Analysis

Our findings concluded that the Shannon Entropy and Minimum Entropy, our measures or randomness, was not meaningfully correlated with either of these observables, but there was a somewhat consistent dependence on the number of qubits that might warrent further investigation with larger systems of qubits. This is of course assuming that maximally entangled systems of qubits can act as random number generators which seems to be possible as the probability of a |1> or |0> state is equally likely within any subset of n maximally entangled qubits. Given that the number of shots used was 5000, a number only possible given our implementation of AWS DM1 noisy quantum simulator (limited budget), the convergence to a uniform distribution might be enabled by the law of large numbers. **Convergence to maximized shannon entropy is likely slower with larger quantum channel error.**  

## Conclusions

Given our analysis, the distinguishability of quantum computers based upon random sequences of numbers is not possible on the basis of the number of maximally entangled qubits and/or the quantum channel error for sufficiently large n (a requirement which is easily achievable).

## Usage

Download the repository and input the following command into your terminal in your preferred coding environment: 

```
pip install -r requirements.txt
```
or 
```
conda install --yes --file requirements. txt
```

run entropy.py to replicate our graphs. To obtain new data, run DataCollectionDiffP or DataCollectionDiffN depending on whether you want n-qubit dependence or channel error respectively. "main.py" is not implemented but can be used to calculate the "categorical-crossentropy" which is a general measure of entropy. 
