<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    displayAlign: "center"
});
</script>

Non-separable set에 대한 support vector machine 문제는 다음과 같다.

>$$
>\begin{align}
>    &\min\_{\beta, \beta\_0, \xi} &&{\frac{1}{2} \|\beta\|\_2^2 + C\sum\_{i=1}^n \xi\_i} \\\\
>    &\text{subject to} &&{\xi\_i \ge 0, \quad i = 1, \dots, n}\\\\
>    & && y\_i (x\_i^T \beta + \beta\_0) \ge 1 - \xi\_i, \quad i = 1, \dots, n,\\\\
>&\text{given } y \in \\{-1, 1\\}^n \text{ and } X \in \mathbb{R}^{n \times p}.
>\end{align}
>$$

주어진 문제의 두 inequality constraints에 대한 0 이상의 Lagrange multipliers를 각각 $v^\star, w^\star$이라 할때, Lagrangian function은 다음과 같다.
>$L(\beta, \beta\_0, \xi, v^\star, w^\star) = \frac{1}{2} \|\beta\|\_2^2 + C\sum\_{i=1}^n \xi\_i - \sum\_{i=1}^n v\_i^\star \xi\_i + \sum\_{i=1}^n w\_i^\star (1 - \xi\_i - y\_i ( x\_i^T \beta + \beta_0))$

위 Lagrangian function을 이용하여 이 문제가 KKT stationarity condition을 만족하게 하는 다음의 조건들을 구할 수 있다. (Lagrangian function을 $\beta, \beta\_0, \xi$에 대해 각각 미분하여 0이 되는 조건을 유도)
>$$
>0 = \sum\_{i=1}^n w\_i^\star y\_i, \quad \beta = \sum\_{i=1}^n w\_i^\star y\_i x\_i, \quad w^\star = C \cdot 1 - v^\star
>$$

또한 두 개의 inequality constraints에 대한 complementary slackness는 다음과 같다.
> $$
> v\_i^\star \xi\_i = 0, \quad w_i^\star (1 - \xi\_i - y\_i (x\_i^T \beta + \beta\_0)) =0, \quad 1 = 1, \dots, n.
> $$

즉, 최적점(optimality)에서 $\beta^\star = \sum\_{i=1}^n w\_i^\star y\_i x\_i$를 만족하며, $y\_i (x\_i^T \beta^\star + \beta\_0^\star) = 1 - \xi\_i^\star$일때 $w\_i^\star$는 nonzero가 되는데, 이런 지점  i를 **support points**라고 부른다.

* 어떤 support point i에 대해, $\xi\_i^\star = 0$이면 $x_i$는 hyperplane 위에 위치하게 되며 이때 $w\_i^\star \in (0, C]$이다.
* 어떤 support point i에 대해, $\xi\_i^\star \neq 0$이면 $x_i$는  hyperplane의 반대쪽에 위치하게 되며 이때 $w\_i^\star = C$다.
<center>
![](https://wikidocs.net/images/page/20962/svm.png)<br/>
**[Fig1] Illustration of support points with $\xi^\star \neq 0$  [3]**
</center>

SVM문제를 최적화 하기 전, non-support points를 걸러내는데 위 방법을 이용할 수 있다 (non-support points를 걸러냄으로써 계산 효율을 높일 수 있다). 사실 KKT conditions는 이 문제의 solution을 도출하기 위한 직접적인 역할을 하지는 않지만, 우리는 이를 통해 SVM 문제를 더 잘 이해하기 위한 직관을 얻을 수 있다 [3].