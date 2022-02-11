# Decision Tree Regressor and Classifier
- Implementation of Decision trees from scratch.
- Performance is close to implementation of scikit-learn.


### Example

```python

N = 30
P = 5

# dtype can be category for discrete type, or float for continuous type
X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size = N),  dtype="category")
criterias = ['information_gain', 'gini_index']:
criterion = criterias[0]
tree = DecisionTree(criterion=criteria) 
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
```


### Time Complexity Analysis

### Real Input Real Output

![](images/experiments_(R,R).png)

![](images/experiments_(D,D).png)

![](images/experiments_(D,R).png)

![](images/experiments_(R,D).png)


#### Fit Time
Theoretical time complexity for fitting a decision tree = $O(mn\log n)$


At constant N, time increases linearly with respect to M ($O(m)$), and matches the theoretical complexity, as seen in the above plots.

At constant M, the time appears to increase linearly with respect to N, which is similar to the $O(n\log n)$ graph. At low values of N, the nlogn graph appears almost linear. Furthermore, due to computational constraints, the graph was evaluated at a finite number of points. If we evaluate at more points, we may be able to observe a better agreement with theory.

#### Predict Time
The predict time complexity is $O(N*depth)$ since we need to traverse the deepest leaf in the worst case. However, since we are maintaining the depth as constant in our experiments, we expect the variation with the number of samples (N) to be linear as each sample has to be classified separately. Furthermore, we also expect the variation with the number of features (M) to be roughly constant, since we do not iterate over the features, but rather use them as criteria for splitting. All of the above are observed in the plots, barring occasional arbitrary deviations.


