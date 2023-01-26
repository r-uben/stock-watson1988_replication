


class WrongInput(Exception):
    def __init__(self,  input_type: type, 
                        right_input_type: type):
        self.input_type = self.take_type(input_type)
        self.right_input_type = self.take_type(right_input_type)

    def __str__(self):
        return self.input_type + ' is not the right type. It should be ' + self.right_input_type

    def take_type(self,my_type):
        return str(my_type).replace("class",'').replace("<","").replace(">",'').strip()[1:-1]