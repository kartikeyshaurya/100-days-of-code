#day 3
# object oriented programming revisted

# CLASSES , OBJECTS , INHERITANCE

# scope and namespace
def scope_test():
    def do_local():
        spam = "local spam"

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam"

    def do_global():
        global spam
        spam = "global spam"

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)




class fruits(object):

    def __init__(self,name , color, shape, taste, productions):
        self.name = name
        self.color = color
        self.taste = taste
        self.shape = shape
        self.productions = productions

    def description(self):
        print(self.color, self.taste, self.shape )

lemon = fruits("lemon", "yellow", "round", "sour", "indian")

print(lemon.description())

# codes for how to be experts in python


