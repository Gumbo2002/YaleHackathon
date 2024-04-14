from braket.circuits import Circuit
from braket.aws import AwsDevice
#Depracated method for our purposes
'''
class TwoBitCircuit():
    def __init__(self, code, control, act):
        self.__name__ = code
        self.control = control
        self.act = act
        self.circuit = Circuit()
        self._activate_bell_state()
    
        if code == 'a0b0':
            self._apply_a0b0()
        elif code == 'a0b1':
            self._apply_a0b1()
        elif code == 'a1b0':
            self._apply_a1b0()
        elif code == 'a1b1':
            self._apply_a1b1()
        else:
            raise ValueError('Wrong code.')
            
    
    def _activate_bell_state(self):
        self.circuit = Circuit()
        self.circuit.h(self.control).cnot(self.control, self.act)
        return 
        
    def _apply_a0b0(self):
        self.circuit.s(self.act).h(self.act).t(self.act).h(self.act)
        return 
    
    def _apply_a0b1(self):
        self.circuit.s(self.act).h(self.act).ti(self.act).h(self.act)
        return 
    
    def _apply_a1b0(self):
        self.circuit.h(self.control)
        self.circuit.s(self.act).h(self.act).t(self.act).h(self.act)
        return 
    
    def _apply_a1b1(self):
        self.circuit.h(self.control)
        self.circuit.s(self.act).h(self.act).ti(self.act).h(self.act)
        return 
'''
class TwoBitCircuit_noCode:
    def __init__(self, control, act):
        self.control = control
        self.act = act
        self.circuit = Circuit()
        self._activate_bell_state()
            
    def _activate_bell_state(self):
        # Initialize maximally entangled state for two qubits
        self.circuit.h(self.control).cnot(self.control, self.act)
        return 

    
class ThreeBitCircuit_noCode:
    def __init__(self):
        self.circuit = Circuit()
        self._activate_bell_state()
    
    def _activate_bell_state(self):
        self.circuit.cnot(2, 0).cnot(2, 1).h(2)
        # This is provided by reference provided here: https://arxiv.org/pdf/1003.3142.pdf
        return 
        # Initialize maximally entangled state for three qubits
        
class FourBitCircuit_noCode:
    def __init__(self, YES_I=True):
        self.circuit = Circuit()
        self._activate_bell_state()
    
    def _activate_bell_state(self):
        self.circuit.cnot(3, 0).cnot(3, 2).h(3).cnot(1, 0).h(1)
        # This is provided by reference provided here: https://arxiv.org/pdf/1003.3142.pdf
        return 
        # Initialize maximally entangled state for four qubits.
        
        
class FiveBitCircuit_noCode:
    def __init__(self, YES_I=True):
        self.circuit = Circuit()
        self._activate_bell_state()
    
    def _activate_bell_state(self):
        # Initialize maximally entangled state for four qubits.
        # This is provided by reference provided here: https://arxiv.org/pdf/1003.3142.pdf
        self.circuit.cnot(3, 2).cnot(4, 1).cnot(1, 0).h(4).cnot(2, 1).h(2) 
        return 
    
class SixBitCircuit_noCode:
    def __init__(self, YES_I=True):
        self.circuit = Circuit()
        self._activate_bell_state()
    
    def _activate_bell_state(self):
        # Initialize maximally entangled state for four qubits.
        # This is provided by reference provided here: https://arxiv.org/pdf/1003.3142.pdf
        self.circuit.cnot(2, 1).cnot(4, 1).h(1).cnot(4, 3).h(4).cnot(5, 2).cnot(3, 0).cnot(5, 4).cnot(3, 0).cnot(5, 4).h(5).cnot(3, 2).h(3).cnot(1, 0).h(1)
        return 