from numpy import exp, array, random, dot
#Clase que crea la red neuronal de 9 entradas por 1 salida
class NeuralNetwork9x1():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((9, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
#Clase que crea la red neuronal de 8 entradas por 1 salida
class NeuralNetwork8x1():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((8, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

    # Inicialice una red neuronal de una sola neurona. Quizas no sea
    # propiamente una red...
    neural_network_1 = NeuralNetwork9x1()
    neural_network_2 = NeuralNetwork9x1()
    neural_network_3 = NeuralNetwork9x1()
    neural_network_4 = NeuralNetwork8x1()

    print("Pesos sinapticos iniciales generados aleatoriamente: ")
    print(neural_network_1.synaptic_weights," pesos para 1")
    print(neural_network_2.synaptic_weights," pesos para 2")
    print(neural_network_3.synaptic_weights," pesos para 3")
    print(neural_network_4.synaptic_weights," pesos para 4")

    # El conjunto de entrenamiento. Tenemos 4 ejemplos, cada uno consistente
    # de tres valores de entrada con su respectiva salida (una salida)
    training_set_inputs_1 = array([[0, 0, 1, 0, 0, 0, 1, 0, 1],#A
                                   [1, 1, 1, 1, 0, 1, 0, 0, 0],#B
                                   [1, 1, 1, 1, 1, 1, 0, 0, 0],#C
                                   [1, 1, 1, 1, 0, 1, 0, 0, 0],#D
                                   [1, 1, 1, 1, 1, 1, 0, 0, 0],#E
                                   [1, 1, 1, 1, 1, 1, 0, 0, 0],#F
                                   [0, 1, 1, 1, 1, 1, 0, 0, 0],#G
                                   [1, 0, 0, 0, 1, 1, 0, 0, 0],#H
                                   [1, 1, 1, 1, 1, 0, 0, 1, 0],#I
                                   [1, 1, 1, 1, 1, 0, 0, 0, 0],#J
                                   [1, 0, 0, 0, 1, 1, 0, 0, 1],#K
                                   [1, 0, 0, 0, 0, 1, 0, 0, 0],#L
                                   [1, 0, 0, 0, 1, 1, 1, 0, 1],#M
                                   [1, 0, 0, 0, 1, 1, 1, 0, 0],#N
                                   [0, 1, 1, 1, 0, 1, 0, 0, 0],#O
                                   [0, 1, 1, 1, 0, 1, 0, 0, 0]])#P

    training_set_inputs_2 = array([ [0, 1, 0, 0, 0, 1, 1, 0, 0],#A
                                    [1, 1, 0, 0, 0, 1, 1, 1, 1],#B
                                    [0, 1, 0, 0, 0, 0, 1, 0, 0],#C
                                    [1, 1, 0, 0, 0, 1, 1, 0, 0],#D
                                    [0, 1, 0, 0, 0, 0, 1, 1, 1],#E
                                    [0, 1, 0, 0, 0, 0, 1, 1, 1],#F
                                    [0, 1, 0, 0, 0, 0, 0, 1, 1],#G
                                    [1, 1, 0, 0, 0, 1, 1, 1, 1],#H
                                    [0, 0, 0, 1, 0, 0, 0, 0, 1],#I
                                    [1, 0, 0, 0, 0, 1, 0, 0, 0],#J
                                    [0, 1, 0, 1, 0, 0, 1, 1, 0],#K
                                    [0, 1, 0, 0, 0, 0, 1, 0, 0],#L
                                    [1, 1, 0, 1, 0, 1, 1, 0, 0],#M
                                    [1, 1, 1, 0, 0, 1, 1, 0, 1],#N
                                    [1, 1, 0, 0, 0, 1, 1, 0, 0],#O
                                    [1, 1, 0, 0, 0, 1, 1, 1, 1]])#P

    training_set_inputs_3 = array([[0, 1, 1, 1, 1, 1, 1, 1, 0],#A
                                   [1, 0, 1, 0, 0, 0, 1, 1, 0],#B
                                   [0, 0, 1, 0, 0, 0, 0, 1, 0],#C
                                   [0, 1, 1, 0, 0, 0, 1, 1, 0],#D
                                   [1, 0, 1, 0, 0, 0, 0, 1, 0],#E
                                   [1, 0, 1, 0, 0, 0, 0, 1, 0],#F
                                   [1, 1, 1, 0, 0, 0, 1, 1, 0],#G
                                   [1, 1, 1, 0, 0, 0, 1, 1, 0],#H
                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],#I
                                   [0, 1, 1, 0, 0, 0, 1, 1, 0],#J
                                   [0, 0, 1, 1, 1, 0, 0, 1, 0],#K
                                   [0, 0, 1, 0, 0, 0, 0, 1, 0],#L
                                   [0, 1, 1, 0, 0, 0, 1, 1, 0],#M
                                   [0, 1, 1, 0, 1, 0, 1, 1, 0],#N
                                   [0, 1, 1, 0, 0, 0, 1, 1, 0],#O
                                   [1, 0, 1, 0, 0, 0, 0, 1, 0]])#P

    training_set_inputs_4 = array([[0, 0, 1, 1, 0, 0, 0, 1],#A
                                   [0, 0, 1, 1, 1, 1, 1, 0],#B
                                   [0, 0, 0, 1, 1, 1, 1, 1],#C
                                   [0, 0, 1, 1, 1, 1, 1, 0],#D
                                   [0, 0, 0, 1, 1, 1, 1, 1],#E
                                   [0, 0, 0, 1, 0, 0, 0, 0],#F
                                   [0, 0, 1, 0, 1, 1, 1, 0],#G
                                   [0, 0, 1, 1, 0, 0, 0, 1],#H
                                   [1, 0, 0, 1, 1, 1, 1, 1],#I
                                   [0, 0, 1, 0, 1, 1, 1, 0],#J
                                   [0, 1, 0, 1, 0, 0, 0, 1],#K
                                   [0, 0, 0, 1, 1, 1, 1, 0],#L
                                   [0, 0, 1, 1, 0, 0, 0, 1],#M
                                   [0, 1, 1, 1, 0, 0, 1, 1],#N
                                   [0, 0, 1, 0, 1, 1, 1, 0],#O
                                   [0, 0, 0, 1, 0, 0, 0, 0]])#P

    training_set_outputs_1 = array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]).T
    training_set_outputs_2 = array([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]).T
    training_set_outputs_3 = array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]).T
    training_set_outputs_4 = array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]]).T

    # Entrene a la red neuronal usando un conjunto de entrenamiento.
    # lo iteramos 10,000 veces, haciendo pequenos ajustes de pesos en cada iteracion
    neural_network_1.train(training_set_inputs_1, training_set_outputs_1, 7000)
    neural_network_2.train(training_set_inputs_2, training_set_outputs_2, 7000)
    neural_network_3.train(training_set_inputs_3, training_set_outputs_3, 7000)
    neural_network_4.train(training_set_inputs_4, training_set_outputs_4, 7000)

    print("Nuevos pesos sinapticos despues del entremaniento: ")
    print(neural_network_1.synaptic_weights," pesos para 1")
    print(neural_network_2.synaptic_weights," pesos para 2")
    print(neural_network_3.synaptic_weights," pesos para 3")
    print(neural_network_4.synaptic_weights," pesos para 4")

    #Convierte la salida punto flotante a binario 1 o 0, eje: 0.65 = 1
    def to_binary(network):
        if network <= 0.5: return 0
        else: return 1
        
    #Convierte las salidas de los diferentes perceptrones a una letra
    #Las entradas tienen que ser en binario
    def to_letter(network_1, network_2, network_3, network_4):
        letter = ""+str(network_1)+str(network_2)+str(network_3)+str(network_4)
        return {
            "0000": "A", "0001": "B", "0010": "C", "0011": "D", "0100": "E", "0101": "F",
            "0110": "G", "0111": "H", "1000": "I", "1001": "J", "1010": "K", "1011": "L",
            "1100": "M", "1101": "N", "1111": "P"
        }[letter]

    # Pruebe la red neuronal con una situacion desconocida.
    print("Considerando las entradas  -> ?: ")
    network_1 = neural_network_1.think(array([1, 1, 1, 1, 1, 1, 0, 0, 0]))
    network_2 = neural_network_2.think(array([0, 1, 0, 0, 0, 0, 1, 0, 0]))
    network_3 = neural_network_3.think(array([0, 0, 1, 0, 0, 0, 0, 1, 0]))
    network_4 = neural_network_4.think(array([0, 0, 0, 1, 1, 1, 1, 1]))

    #Llama a la funcion to_binary
    print(to_binary(network_1), "Para 1", network_1)
    print(to_binary(network_2), "Para 2", network_2)
    print(to_binary(network_3), "Para 3", network_3)
    print(to_binary(network_4), "Para 4", network_4)

    #Llama a la funcion to_letter
    print("Letra : ",to_letter(to_binary(network_1), to_binary(network_2), to_binary(network_3), to_binary(network_4)))


