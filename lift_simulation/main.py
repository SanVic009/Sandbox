"""
This file will have the implementation of my lift simulation
"""

# Defining max floors
MAX = 6

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

class Queue:
    def __init__(self):
        self.front = None
        self.rear = None
        self.length = 0

    def enqueue(self, index, val):

        if index < 0 or index >= self.length:
            print("Invalid index while enqueue")
            return

        temp = self.front
        newNode = Node(val)


current_floor = 0
