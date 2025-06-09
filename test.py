import dectree

print("Decision Tree Prediction")
tree = dectree.DecisionTree()
root = dectree.Split(0, 10, 10)
root.left = dectree.Split(0, 5, 5)
root.right = dectree.Split(0, 15, 15)
root.left.left = dectree.Split(0, 2.5, 2.5)
root.left.right = dectree.Split(0, 7.5, 7.5)
root.right.left = dectree.Split(0, 12.5, 12.5)
root.right.right = dectree.Split(0, 17.5, 17.5)

tree.set_root(root)

for i in range(-10, 30):
    predicted = tree.predict(i)
    print(f"In: {i} Out: {predicted}")

'''
         10
    5          15
2.5   7.5  12.5   17.5
    '''
