// Extended documentation for each algorithm
export interface AlgorithmDoc {
    id: string;
    name: string;
    fullName: string;
    category: string;

    // Theory
    introduction: string;
    mathematicalBackground: {
        title: string;
        content: string;
        formula?: string;
    }[];

    // Implementation
    pseudocode: string;
    complexity: {
        time: string;
        space: string;
    };

    // Practical
    whenToUse: string[];
    whenNotToUse: string[];
    tips: string[];

    // Examples
    exampleDataset: string;
    exampleOutput: string;

    // References
    references: {
        title: string;
        url: string;
    }[];
}

export const algorithmDocs: Record<string, AlgorithmDoc> = {
    'k-means': {
        id: 'k-means',
        name: 'K-Means',
        fullName: 'K-Means Clustering Algorithm',
        category: 'Clustering',

        introduction: `K-Means is one of the most popular and widely used unsupervised machine learning algorithms for clustering. Developed by Stuart Lloyd in 1957 (published in 1982) and later by James MacQueen in 1967, K-Means partitions n observations into k clusters where each observation belongs to the cluster with the nearest mean (centroid).

The algorithm is iterative and converges to a local optimum, making it efficient for large datasets. It's particularly effective when clusters are spherical, equally sized, and well-separated.`,

        mathematicalBackground: [
            {
                title: 'Objective Function',
                content: 'K-Means minimizes the Within-Cluster Sum of Squares (WCSS), also known as inertia. The goal is to minimize the total distance between each point and its assigned cluster centroid.',
                formula: 'J = Σᵢ₌₁ⁿ Σⱼ₌₁ᵏ wᵢⱼ ||xᵢ - μⱼ||²'
            },
            {
                title: 'Centroid Calculation',
                content: 'After assigning points to clusters, each centroid is recalculated as the mean of all points in that cluster.',
                formula: 'μⱼ = (1/|Cⱼ|) Σₓ∈Cⱼ x'
            },
            {
                title: 'Distance Metric',
                content: 'Standard K-Means uses Euclidean distance to measure similarity between points and centroids.',
                formula: 'd(x, μ) = √(Σᵢ(xᵢ - μᵢ)²)'
            }
        ],

        pseudocode: `Algorithm: K-Means Clustering
----------------------------------------
Input: Dataset X, number of clusters K
Output: Cluster assignments, centroids

1. INITIALIZE K centroids randomly from X
2. REPEAT:
   a. ASSIGN each point to nearest centroid
      For each point xᵢ:
        cluster[i] = argminⱼ ||xᵢ - μⱼ||²
   
   b. UPDATE centroids
      For each cluster j:
        μⱼ = mean of all points in cluster j
   
   c. CALCULATE total WCSS (inertia)
   
3. UNTIL centroids don't change (convergence)
   OR max iterations reached

4. RETURN cluster assignments and centroids`,

        complexity: {
            time: 'O(n × k × i × d) where n=points, k=clusters, i=iterations, d=dimensions',
            space: 'O(n × d + k × d) for storing data and centroids'
        },

        whenToUse: [
            'When you have a rough idea of the number of clusters',
            'Data is roughly spherical/convex in shape',
            'Clusters are of similar size',
            'You need fast, scalable clustering',
            'Working with numerical continuous data'
        ],

        whenNotToUse: [
            'Unknown number of clusters (consider DBSCAN)',
            'Non-spherical or irregular cluster shapes',
            'Clusters of very different sizes/densities',
            'Data has many outliers',
            'Categorical data without proper encoding'
        ],

        tips: [
            'Use the Elbow Method or Silhouette Score to find optimal K',
            'Run multiple times with different initializations',
            'Use K-Means++ initialization for better starting points',
            'Normalize/standardize features before clustering',
            'Check cluster sizes - very small clusters may indicate overfitting'
        ],

        exampleDataset: `sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
7.0,3.2,4.7,1.4
6.4,3.2,4.5,1.5
6.3,3.3,6.0,2.5
5.8,2.7,5.1,1.9`,

        exampleOutput: `{
  "clusters": [0, 0, 1, 1, 2, 2],
  "centroids": [
    [5.0, 3.25, 1.4, 0.2],
    [6.7, 3.2, 4.6, 1.45],
    [6.05, 3.0, 5.55, 2.2]
  ],
  "inertia": 2.34,
  "silhouetteScore": 0.82
}`,

        references: [
            { title: 'Original Paper by MacQueen (1967)', url: 'https://www.cs.cmu.edu/~guestrin/Class/10701/schedule/kmeans.pdf' },
            { title: 'K-Means++ Initialization', url: 'http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf' },
            { title: 'Scikit-learn Documentation', url: 'https://scikit-learn.org/stable/modules/clustering.html#k-means' }
        ]
    },

    'naive-bayes': {
        id: 'naive-bayes',
        name: 'Naive Bayes',
        fullName: 'Gaussian Naive Bayes Classifier',
        category: 'Classification',

        introduction: `Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with a strong (naive) independence assumption between features. Despite this simplifying assumption being rarely true in real-world data, Naive Bayes often performs remarkably well in practice.

The algorithm is particularly popular for text classification tasks like spam detection and sentiment analysis. It's fast, requires minimal training data, and scales well to high-dimensional datasets.`,

        mathematicalBackground: [
            {
                title: "Bayes' Theorem",
                content: 'The foundation of Naive Bayes is Bayes\' theorem, which describes the probability of a class given the observed features.',
                formula: 'P(C|X) = P(X|C) × P(C) / P(X)'
            },
            {
                title: 'Naive Independence Assumption',
                content: 'The "naive" assumption is that all features are conditionally independent given the class. This simplifies the likelihood calculation.',
                formula: 'P(X|C) = P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)'
            },
            {
                title: 'Gaussian Distribution',
                content: 'For continuous features, Gaussian Naive Bayes assumes features follow a normal distribution within each class.',
                formula: 'P(xᵢ|C) = (1/√(2πσ²)) × exp(-(xᵢ - μ)²/(2σ²))'
            }
        ],

        pseudocode: `Algorithm: Gaussian Naive Bayes
----------------------------------------
TRAINING Phase:
1. For each class c in classes:
   a. Calculate P(c) = count(c) / total_samples
   b. For each feature i:
      - Calculate mean μᵢc for class c
      - Calculate variance σ²ᵢc for class c

PREDICTION Phase:
For each test sample x:
1. For each class c:
   a. Start with log(P(c))
   b. For each feature xᵢ:
      - Add log(P(xᵢ|c)) using Gaussian PDF
   c. Store total log-probability
2. Return class with highest log-probability`,

        complexity: {
            time: 'O(n × d) for training, O(k × d) for prediction',
            space: 'O(k × d) for storing means and variances per class'
        },

        whenToUse: [
            'Text classification (spam, sentiment, topic)',
            'Quick baseline model needed',
            'Features are roughly independent',
            'Limited training data available',
            'Real-time prediction required'
        ],

        whenNotToUse: [
            'Features are highly correlated',
            'Relationship between features matters',
            'You need probability calibration',
            'Complex decision boundaries required',
            'Numerical precision is critical'
        ],

        tips: [
            'Use Laplace smoothing to handle zero probabilities',
            'Consider Multinomial NB for text/count data',
            'Try Complement NB for imbalanced datasets',
            'Log-transform probabilities to avoid underflow',
            'Feature selection can improve performance'
        ],

        exampleDataset: `age,income,student,credit_rating,buys_computer
30,high,no,fair,no
30,high,no,excellent,no
40,high,no,fair,yes
50,medium,no,fair,yes
50,low,yes,fair,yes`,

        exampleOutput: `{
  "prediction": "yes",
  "probabilities": {
    "yes": 0.73,
    "no": 0.27
  },
  "accuracy": 85.5,
  "confusionMatrix": [[12, 3], [2, 23]]
}`,

        references: [
            { title: 'Naive Bayes Tutorial', url: 'https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html' },
            { title: 'Text Classification with NB', url: 'https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html' },
            { title: 'Scikit-learn Naive Bayes', url: 'https://scikit-learn.org/stable/modules/naive_bayes.html' }
        ]
    },

    'pca': {
        id: 'pca',
        name: 'PCA',
        fullName: 'Principal Component Analysis',
        category: 'Dimensionality Reduction',

        introduction: `Principal Component Analysis (PCA) is a statistical technique for dimensionality reduction that transforms data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they capture from the original data.

Invented by Karl Pearson in 1901 and developed independently by Harold Hotelling in the 1930s, PCA is fundamental in data science for visualization, noise reduction, feature extraction, and preprocessing before other ML algorithms.`,

        mathematicalBackground: [
            {
                title: 'Covariance Matrix',
                content: 'PCA starts by computing the covariance matrix of the standardized data, which captures the relationships between all pairs of features.',
                formula: 'Σ = (1/(n-1)) × XᵀX'
            },
            {
                title: 'Eigendecomposition',
                content: 'The covariance matrix is decomposed into eigenvectors (directions) and eigenvalues (variance explained). Eigenvectors become the principal components.',
                formula: 'Σv = λv'
            },
            {
                title: 'Projection',
                content: 'Original data is projected onto the top k eigenvectors to get the reduced representation.',
                formula: 'Z = X × Wₖ'
            },
            {
                title: 'Variance Explained',
                content: 'The proportion of variance explained by each component is the ratio of its eigenvalue to the sum of all eigenvalues.',
                formula: 'Var(PCᵢ) = λᵢ / Σⱼλⱼ'
            }
        ],

        pseudocode: `Algorithm: Principal Component Analysis
----------------------------------------
Input: Data matrix X (n × d), components k
Output: Reduced data Z (n × k)

1. STANDARDIZE data (zero mean, unit variance)
   X_std = (X - mean(X)) / std(X)

2. COMPUTE covariance matrix
   Σ = (1/(n-1)) × X_stdᵀ × X_std

3. COMPUTE eigenvalues and eigenvectors
   eigenvalues, eigenvectors = eigen(Σ)

4. SORT eigenvectors by eigenvalue (descending)

5. SELECT top k eigenvectors → W_k

6. PROJECT data onto new axes
   Z = X_std × W_k

7. RETURN Z, variance explained`,

        complexity: {
            time: 'O(d³) for eigendecomposition or O(n×d×k) for iterative methods',
            space: 'O(d²) for covariance matrix'
        },

        whenToUse: [
            'High-dimensional data (d > 50 features)',
            'Need to visualize data in 2D/3D',
            'Remove multicollinearity before regression',
            'Reduce noise in data',
            'Speed up other ML algorithms'
        ],

        whenNotToUse: [
            'Features have different scales (must standardize first)',
            'Non-linear relationships dominate (use t-SNE, UMAP)',
            'Interpretability of original features is crucial',
            'Sparse data (consider Sparse PCA)',
            'Need to preserve distances (consider MDS)'
        ],

        tips: [
            'Always standardize data before PCA',
            'Use scree plot to decide number of components',
            'Aim for 80-95% variance explained',
            'Check loadings to interpret components',
            'Incremental PCA for very large datasets'
        ],

        exampleDataset: `feature1,feature2,feature3,feature4,feature5
2.5,2.4,3.5,1.2,4.1
0.5,0.7,0.8,0.3,1.2
2.2,2.9,3.1,1.1,3.8
1.9,2.2,2.5,0.9,3.2
3.1,3.0,4.2,1.5,4.8`,

        exampleOutput: `{
  "transformedData": [
    [4.23, -0.12],
    [-3.45, 0.08],
    [3.89, 0.34],
    ...
  ],
  "explainedVariance": [72.4, 18.3],
  "totalVarianceExplained": 90.7,
  "components": [
    [0.45, 0.42, 0.48, 0.35, 0.52],
    [-0.23, 0.67, -0.12, 0.45, -0.32]
  ]
}`,

        references: [
            { title: 'PCA Tutorial by Jonathon Shlens', url: 'https://arxiv.org/abs/1404.1100' },
            { title: 'StatQuest PCA Explanation', url: 'https://www.youtube.com/watch?v=FgakZw6K1QQ' },
            { title: 'Scikit-learn PCA', url: 'https://scikit-learn.org/stable/modules/decomposition.html#pca' }
        ]
    },

    'logistic-regression': {
        id: 'logistic-regression',
        name: 'Logistic Regression',
        fullName: 'Logistic Regression Classifier',
        category: 'Classification',

        introduction: `Despite its name, Logistic Regression is a classification algorithm, not regression. It models the probability that an instance belongs to a particular class using the logistic (sigmoid) function, making it ideal for binary classification.

Developed in the 19th century for demographic studies, it became a staple in machine learning due to its simplicity, interpretability, and effectiveness. It's often the first algorithm tried for classification problems.`,

        mathematicalBackground: [
            {
                title: 'Logistic Function (Sigmoid)',
                content: 'The sigmoid function maps any real number to a probability between 0 and 1.',
                formula: 'σ(z) = 1 / (1 + e⁻ᶻ)'
            },
            {
                title: 'Linear Combination',
                content: 'The input to the sigmoid is a linear combination of features and weights.',
                formula: 'z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ'
            },
            {
                title: 'Log-Odds (Logit)',
                content: 'The log-odds of the positive class is linear in the features.',
                formula: 'log(p/(1-p)) = β₀ + βᵀx'
            },
            {
                title: 'Loss Function',
                content: 'Binary cross-entropy is used to measure prediction error.',
                formula: 'L = -Σ[yᵢlog(pᵢ) + (1-yᵢ)log(1-pᵢ)]'
            }
        ],

        pseudocode: `Algorithm: Logistic Regression
----------------------------------------
TRAINING (Gradient Descent):
1. Initialize weights β to zeros or small random
2. For each iteration:
   a. Compute predictions: p = σ(Xβ)
   b. Compute gradient: ∇ = Xᵀ(p - y)
   c. Update weights: β = β - α∇
3. Repeat until convergence

PREDICTION:
1. Compute z = β₀ + βᵀx
2. Compute p = σ(z)
3. If p ≥ 0.5, predict class 1
   Else predict class 0`,

        complexity: {
            time: 'O(n × d × i) for training with gradient descent',
            space: 'O(d) for storing weights'
        },

        whenToUse: [
            'Binary classification problems',
            'Need probability outputs',
            'Linear decision boundary is sufficient',
            'Interpretability is important',
            'Baseline model for comparison'
        ],

        whenNotToUse: [
            'Complex non-linear relationships',
            'Multi-class without modification',
            'Features are highly correlated (use regularization)',
            'Very high-dimensional sparse data',
            'Outliers significantly affect results'
        ],

        tips: [
            'Use L1/L2 regularization to prevent overfitting',
            'Standardize features for faster convergence',
            'Check for multicollinearity using VIF',
            'Examine coefficients for feature importance',
            'Use ROC-AUC for imbalanced datasets'
        ],

        exampleDataset: `hours_studied,hours_slept,passed
5,8,1
3,6,0
8,7,1
2,4,0
6,8,1`,

        exampleOutput: `{
  "coefficients": {
    "intercept": -4.2,
    "hours_studied": 0.85,
    "hours_slept": 0.32
  },
  "accuracy": 87.5,
  "precision": 0.89,
  "recall": 0.86,
  "f1Score": 0.87
}`,

        references: [
            { title: 'Logistic Regression Tutorial', url: 'https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf' },
            { title: 'Andrew Ng ML Course', url: 'https://www.coursera.org/learn/machine-learning' },
            { title: 'Scikit-learn Logistic Regression', url: 'https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression' }
        ]
    },

    'linear-regression': {
        id: 'linear-regression',
        name: 'Linear Regression',
        fullName: 'Ordinary Least Squares Linear Regression',
        category: 'Regression',

        introduction: `Linear Regression is the foundational algorithm for regression tasks. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation that minimizes the sum of squared residuals.

Dating back to the early 19th century with Legendre and Gauss, it remains one of the most widely used techniques due to its simplicity, interpretability, and effectiveness when the underlying relationship is approximately linear.`,

        mathematicalBackground: [
            {
                title: 'Model Equation',
                content: 'The model assumes a linear relationship between features and target.',
                formula: 'y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε'
            },
            {
                title: 'Least Squares Objective',
                content: 'Find coefficients that minimize the sum of squared errors.',
                formula: 'min Σᵢ(yᵢ - ŷᵢ)² = min ||y - Xβ||²'
            },
            {
                title: 'Normal Equation',
                content: 'Closed-form solution for the optimal coefficients.',
                formula: 'β = (XᵀX)⁻¹Xᵀy'
            },
            {
                title: 'R-squared',
                content: 'Coefficient of determination measuring variance explained.',
                formula: 'R² = 1 - (SS_res / SS_tot)'
            }
        ],

        pseudocode: `Algorithm: Linear Regression
----------------------------------------
Method 1 - Normal Equation:
1. Add column of 1s to X for intercept
2. Compute β = (XᵀX)⁻¹Xᵀy

Method 2 - Gradient Descent:
1. Initialize β to zeros
2. For each iteration:
   a. Compute predictions: ŷ = Xβ
   b. Compute error: e = ŷ - y
   c. Compute gradient: ∇ = (2/n) × Xᵀe
   d. Update weights: β = β - α∇
3. Repeat until convergence

PREDICTION:
ŷ = β₀ + β₁x₁ + ... + βₙxₙ`,

        complexity: {
            time: 'O(n×d² + d³) for normal equation, O(n×d×i) for gradient descent',
            space: 'O(d²) for normal equation, O(d) for gradient descent'
        },

        whenToUse: [
            'Continuous target variable',
            'Linear relationship exists',
            'Interpretability is needed',
            'Quick baseline model',
            'Features are well-scaled'
        ],

        whenNotToUse: [
            'Non-linear relationships',
            'Outliers dominate data',
            'Heteroscedasticity present',
            'Multicollinearity among features',
            'Target is categorical'
        ],

        tips: [
            'Check residual plots for linearity assumptions',
            'Use regularization (Ridge/Lasso) for many features',
            'Transform features if non-linear patterns exist',
            'Handle outliers before fitting',
            'Check VIF for multicollinearity'
        ],

        exampleDataset: `square_feet,bedrooms,age,price
1500,3,10,300000
2000,4,5,450000
1200,2,20,220000
1800,3,8,380000`,

        exampleOutput: `{
  "coefficients": {
    "intercept": 50000,
    "square_feet": 150.5,
    "bedrooms": 25000,
    "age": -2500
  },
  "r_squared": 0.92,
  "adjusted_r_squared": 0.89,
  "rmse": 18500
}`,

        references: [
            { title: 'Linear Regression Deep Dive', url: 'https://www.stat.cmu.edu/~larry/=stat401/lectures/lecture13.pdf' },
            { title: 'OLS Assumptions', url: 'https://www.statisticssolutions.com/assumptions-of-linear-regression/' },
            { title: 'Scikit-learn Linear Regression', url: 'https://scikit-learn.org/stable/modules/linear_model.html' }
        ]
    },

    'dbscan': {
        id: 'dbscan',
        name: 'DBSCAN',
        fullName: 'Density-Based Spatial Clustering of Applications with Noise',
        category: 'Clustering',

        introduction: `DBSCAN is a density-based clustering algorithm that groups together points that are closely packed, marking points in low-density regions as outliers. Unlike K-Means, it doesn't require specifying the number of clusters and can discover clusters of arbitrary shape.

Proposed by Martin Ester et al. in 1996, DBSCAN is particularly useful for spatial data and anomaly detection. It's one of the most cited algorithms in the database/data mining literature.`,

        mathematicalBackground: [
            {
                title: 'Core Point',
                content: 'A point is a core point if it has at least MinPts points within ε distance.',
                formula: '|Nε(p)| ≥ MinPts where Nε(p) = {q ∈ D : dist(p,q) ≤ ε}'
            },
            {
                title: 'Density-Reachable',
                content: 'Point q is density-reachable from p if there is a chain of core points connecting them.',
                formula: 'p →ε q₁ →ε q₂ →ε ... →ε q'
            },
            {
                title: 'Density-Connected',
                content: 'Two points are density-connected if they are both density-reachable from the same core point.',
                formula: '∃o : p ←ε o →ε q'
            }
        ],

        pseudocode: `Algorithm: DBSCAN
----------------------------------------
Input: Dataset D, ε (neighborhood radius), MinPts
Output: Clusters and noise points

1. Mark all points as UNVISITED
2. For each unvisited point p:
   a. Mark p as VISITED
   b. Find neighbors N = points within ε of p
   c. If |N| < MinPts:
      - Mark p as NOISE
   d. Else:
      - Create new cluster C
      - Add p to C
      - For each point q in N:
        - If q is UNVISITED:
          - Mark q VISITED
          - Find neighbors of q
          - If |neighbors| ≥ MinPts:
            - Add neighbors to N
        - If q not in any cluster:
          - Add q to C
3. Return all clusters`,

        complexity: {
            time: 'O(n²) naive, O(n log n) with spatial index like R-tree',
            space: 'O(n) for storing cluster labels and visited status'
        },

        whenToUse: [
            'Unknown number of clusters',
            'Clusters have irregular shapes',
            'Need to detect outliers/anomalies',
            'Spatial or geographic data',
            'Clusters have similar density'
        ],

        whenNotToUse: [
            'Clusters have varying densities',
            'Very high-dimensional data',
            'Need deterministic results for border points',
            'Real-time updating required',
            'ε and MinPts are hard to determine'
        ],

        tips: [
            'Use k-distance graph to find optimal ε',
            'MinPts ≥ dimensions + 1 is a good starting point',
            'Normalize features for meaningful distances',
            'Consider HDBSCAN for varying density clusters',
            'Use spatial indexing for large datasets'
        ],

        exampleDataset: `x,y
1.0,1.0
1.2,0.8
0.9,1.1
5.0,5.0
5.2,4.8
4.9,5.1
10.0,1.0`,

        exampleOutput: `{
  "labels": [0, 0, 0, 1, 1, 1, -1],
  "n_clusters": 2,
  "n_noise": 1,
  "core_samples": [0, 1, 2, 3, 4, 5]
}`,

        references: [
            { title: 'Original DBSCAN Paper', url: 'https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf' },
            { title: 'Choosing ε and MinPts', url: 'https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf' },
            { title: 'Scikit-learn DBSCAN', url: 'https://scikit-learn.org/stable/modules/clustering.html#dbscan' }
        ]
    },

    'svm': {
        id: 'svm',
        name: 'SVM',
        fullName: 'Support Vector Machine',
        category: 'Classification',

        introduction: `Support Vector Machines (SVMs) are powerful supervised learning algorithms that find the optimal hyperplane separating different classes. They maximize the margin between classes, making them robust to overfitting.

Developed by Vladimir Vapnik and colleagues in the 1990s, SVMs can handle high-dimensional data effectively and use the "kernel trick" to tackle non-linear classification problems without explicitly computing transformations.`,

        mathematicalBackground: [
            {
                title: 'Decision Boundary',
                content: 'SVM finds a hyperplane that separates classes with maximum margin.',
                formula: 'wᵀx + b = 0'
            },
            {
                title: 'Margin Maximization',
                content: 'The objective is to maximize the margin (distance between hyperplane and nearest points).',
                formula: 'max 2/||w|| subject to yᵢ(wᵀxᵢ + b) ≥ 1'
            },
            {
                title: 'Kernel Trick',
                content: 'Map data to higher dimensions without explicit computation using kernel functions.',
                formula: 'K(x, y) = φ(x)ᵀφ(y)'
            },
            {
                title: 'Common Kernels',
                content: 'RBF (Gaussian) kernel is most popular for non-linear classification.',
                formula: 'K(x, y) = exp(-γ||x - y||²)'
            }
        ],

        pseudocode: `Algorithm: SVM (Simplified)
----------------------------------------
TRAINING:
1. Transform to dual problem with Lagrange multipliers
2. Solve quadratic programming problem:
   max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
   subject to: αᵢ ≥ 0, Σαᵢyᵢ = 0
3. Find support vectors (αᵢ > 0)
4. Compute w = Σαᵢyᵢxᵢ and b

PREDICTION:
1. For new point x:
   f(x) = sign(Σαᵢyᵢ K(xᵢ, x) + b)
2. Return class based on sign`,

        complexity: {
            time: 'O(n² × d) to O(n³) depending on kernel and solver',
            space: 'O(n × d) for storing support vectors'
        },

        whenToUse: [
            'High-dimensional data (d >> n)',
            'Clear margin of separation exists',
            'Text classification, image recognition',
            'Small to medium datasets',
            'Binary classification or one-vs-rest'
        ],

        whenNotToUse: [
            'Very large datasets (> 100k samples)',
            'Highly noisy data with overlapping classes',
            'Need probability estimates (requires calibration)',
            'Many classes (becomes slow)',
            'Online/incremental learning needed'
        ],

        tips: [
            'Always scale/normalize features',
            'Use cross-validation to tune C and γ',
            'Start with RBF kernel, then try others',
            'Use grid search for hyperparameter tuning',
            'Consider LinearSVC for large datasets'
        ],

        exampleDataset: `feature1,feature2,label
1.0,2.0,1
1.5,1.8,1
2.0,2.2,1
5.0,5.0,-1
5.5,4.8,-1
6.0,5.2,-1`,

        exampleOutput: `{
  "support_vectors": [[2.0, 2.2], [5.0, 5.0]],
  "n_support": [1, 1],
  "accuracy": 95.0,
  "decision_function_values": [2.3, 1.8, 1.2, -1.5, -2.1, -2.8]
}`,

        references: [
            { title: 'Tutorial on SVMs', url: 'https://www.cs.cmu.edu/~guestrin/Class/10701/slides/svm.pdf' },
            { title: 'Kernel Methods', url: 'https://www.cs.columbia.edu/~jebara/4772/tutorials/kernel_tutorial.pdf' },
            { title: 'Scikit-learn SVM', url: 'https://scikit-learn.org/stable/modules/svm.html' }
        ]
    },

    'tf-idf-nb': {
        id: 'tf-idf-nb',
        name: 'TF-IDF + Naive Bayes',
        fullName: 'TF-IDF with Multinomial Naive Bayes',
        category: 'Text Mining',

        introduction: `This combination is the classic approach for text classification. TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors that weight words by their importance, while Multinomial Naive Bayes classifies based on word occurrence probabilities.

Together, they form a powerful, fast, and interpretable baseline for tasks like spam detection, sentiment analysis, and document categorization.`,

        mathematicalBackground: [
            {
                title: 'Term Frequency (TF)',
                content: 'Measures how frequently a term appears in a document.',
                formula: 'TF(t,d) = count(t in d) / total_words(d)'
            },
            {
                title: 'Inverse Document Frequency (IDF)',
                content: 'Measures how important a term is across the entire corpus.',
                formula: 'IDF(t) = log(N / DF(t))'
            },
            {
                title: 'TF-IDF Score',
                content: 'Combines TF and IDF to get final weight.',
                formula: 'TF-IDF(t,d) = TF(t,d) × IDF(t)'
            },
            {
                title: 'Multinomial NB',
                content: 'Probability of class given word counts.',
                formula: 'P(c|d) ∝ P(c) × Πₜ P(t|c)^count(t)'
            }
        ],

        pseudocode: `Algorithm: TF-IDF + Naive Bayes
----------------------------------------
PREPROCESSING:
1. Tokenize documents into words
2. Remove stopwords and punctuation
3. Apply stemming/lemmatization (optional)

TF-IDF VECTORIZATION:
1. Build vocabulary from all documents
2. For each document d:
   For each term t:
     - Compute TF(t,d)
     - Compute IDF(t) across corpus
     - TF-IDF(t,d) = TF × IDF
3. Output: sparse matrix of TF-IDF scores

NAIVE BAYES TRAINING:
1. For each class c:
   - P(c) = docs in c / total docs
   - For each term t:
     P(t|c) = (count of t in c + α) / 
              (total words in c + α×|V|)

PREDICTION:
1. Vectorize new document with TF-IDF
2. For each class c:
   - Compute log P(c) + Σ log P(t|c) × w(t)
3. Return class with highest score`,

        complexity: {
            time: 'O(n × d) for vectorization and training',
            space: 'O(n × |V|) for TF-IDF matrix (sparse)'
        },

        whenToUse: [
            'Text classification tasks',
            'Email spam detection',
            'Sentiment analysis',
            'Document categorization',
            'Fast baseline needed'
        ],

        whenNotToUse: [
            'Word order matters (use RNN/Transformers)',
            'Semantic understanding needed (use embeddings)',
            'Very short texts (few words)',
            'Multilingual without preprocessing',
            'Need to capture context'
        ],

        tips: [
            'Use n-grams (bigrams/trigrams) for context',
            'Experiment with max_features limit',
            'Apply min_df to remove rare words',
            'Consider sublinear TF scaling',
            'Try Complement Naive Bayes for imbalanced data'
        ],

        exampleDataset: `text,category
"Great product, highly recommend",positive
"Terrible quality, waste of money",negative
"Love it! Will buy again",positive
"Broke after one week",negative`,

        exampleOutput: `{
  "prediction": "positive",
  "probabilities": {
    "positive": 0.89,
    "negative": 0.11
  },
  "top_features_positive": ["great", "recommend", "love"],
  "top_features_negative": ["terrible", "waste", "broke"]
}`,

        references: [
            { title: 'TF-IDF Weighting', url: 'https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html' },
            { title: 'Text Classification with NB', url: 'https://web.stanford.edu/~jurafsky/slp3/4.pdf' },
            { title: 'Scikit-learn Text Features', url: 'https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction' }
        ]
    },

    'decision-tree': {
        id: 'decision-tree',
        name: 'Decision Tree',
        fullName: 'Decision Tree Classifier',
        category: 'Classification',

        introduction: `Decision Trees are one of the most interpretable machine learning algorithms. They create a tree-like model where each internal node represents a decision based on a feature, branches represent outcomes, and leaves represent class labels.

The algorithm recursively partitions the data based on feature values that maximize information gain or minimize impurity (Gini). Decision Trees are the foundation for powerful ensemble methods like Random Forest.`,

        mathematicalBackground: [
            {
                title: 'Gini Impurity',
                content: 'Measures the probability of misclassifying a randomly chosen element.',
                formula: 'Gini(D) = 1 - Σᵢ pᵢ²'
            },
            {
                title: 'Information Gain (Entropy)',
                content: 'Measures the reduction in entropy after a split.',
                formula: 'IG(D,A) = H(D) - Σᵥ (|Dᵥ|/|D|) × H(Dᵥ)'
            },
            {
                title: 'Entropy',
                content: 'Measures the impurity or randomness in the data.',
                formula: 'H(D) = -Σᵢ pᵢ log₂(pᵢ)'
            }
        ],

        pseudocode: `Algorithm: Decision Tree (ID3/CART)
----------------------------------------
BUILD_TREE(D, features):
1. If all samples in D have same class:
   Return leaf with that class
2. If no features left or max_depth reached:
   Return leaf with majority class
3. Select best feature A using Gini/IG
4. Create node for feature A
5. For each value v of A:
   a. Create branch for v
   b. Dᵥ = subset of D where A = v
   c. If Dᵥ is empty:
      Add leaf with majority class of D
   d. Else:
      Add subtree BUILD_TREE(Dᵥ, features - A)
6. Return tree`,

        complexity: {
            time: 'O(n × d × log(n)) for training, O(log(n)) for prediction',
            space: 'O(n) for storing the tree'
        },

        whenToUse: [
            'Interpretability is crucial',
            'Mix of numerical and categorical features',
            'Non-linear decision boundaries needed',
            'Feature importance analysis',
            'Quick baseline model'
        ],

        whenNotToUse: [
            'Data is highly linear (use logistic regression)',
            'Need stable predictions (trees are sensitive)',
            'Very high dimensional sparse data',
            'Extrapolation beyond training range needed',
            'Large datasets without pruning'
        ],

        tips: [
            'Use pruning (max_depth, min_samples) to prevent overfitting',
            'Balance classes before training',
            'Use cross-validation to tune hyperparameters',
            'Visualize the tree to gain insights',
            'Consider ensemble methods for better accuracy'
        ],

        exampleDataset: `outlook,temperature,humidity,windy,play
sunny,hot,high,false,no
sunny,hot,high,true,no
overcast,hot,high,false,yes
rainy,mild,high,false,yes`,

        exampleOutput: `{
  "tree": {
    "feature": "outlook",
    "branches": {
      "sunny": {"feature": "humidity", ...},
      "overcast": {"class": "yes"},
      "rainy": {"feature": "windy", ...}
    }
  },
  "accuracy": 92.5,
  "feature_importance": {"outlook": 0.45, "humidity": 0.32, ...}
}`,

        references: [
            { title: 'ID3 Algorithm', url: 'https://hunch.net/~coms-4771/quinlan.pdf' },
            { title: 'CART Algorithm', url: 'https://www.stat.berkeley.edu/~breiman/CART.pdf' },
            { title: 'Scikit-learn Decision Trees', url: 'https://scikit-learn.org/stable/modules/tree.html' }
        ]
    },

    'random-forest': {
        id: 'random-forest',
        name: 'Random Forest',
        fullName: 'Random Forest Ensemble',
        category: 'Classification',

        introduction: `Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It uses bagging (bootstrap aggregating) and random feature selection to reduce overfitting.

Introduced by Leo Breiman in 2001, Random Forest has become one of the most successful and widely used machine learning algorithms due to its excellent performance across many domains.`,

        mathematicalBackground: [
            {
                title: 'Bootstrap Sampling',
                content: 'Each tree is trained on a random sample with replacement.',
                formula: 'Sample size = n (with replacement from n samples)'
            },
            {
                title: 'Random Feature Selection',
                content: 'At each split, only a random subset of features is considered.',
                formula: 'm = √d for classification, m = d/3 for regression'
            },
            {
                title: 'Aggregation',
                content: 'Final prediction is the mode (classification) or mean (regression).',
                formula: 'ŷ = mode(tree₁(x), tree₂(x), ..., treeₙ(x))'
            }
        ],

        pseudocode: `Algorithm: Random Forest
----------------------------------------
TRAINING:
1. For b = 1 to B (number of trees):
   a. Draw bootstrap sample of size n
   b. Grow tree Tᵦ:
      - At each node, select m random features
      - Find best split among m features
      - Split node into two children
      - Repeat until min_samples_leaf
   c. Save tree Tᵦ

PREDICTION:
1. For new sample x:
   a. Get prediction from each tree
   b. Return majority vote (or average)

OUT-OF-BAG ERROR:
1. For each sample, predict using trees 
   that didn't include it in bootstrap
2. Calculate error rate`,

        complexity: {
            time: 'O(B × n × d × log(n)) for training',
            space: 'O(B × tree_size) for storing all trees'
        },

        whenToUse: [
            'Tabular data with mixed feature types',
            'Need robust predictions',
            'Feature importance analysis',
            'Don\'t want to tune many hyperparameters',
            'Classification or regression tasks'
        ],

        whenNotToUse: [
            'Interpretability is crucial (use single tree)',
            'Memory is very limited',
            'Real-time predictions with latency constraints',
            'Deep learning outperforms (images, text)',
            'Data is very sparse'
        ],

        tips: [
            'Use OOB (out-of-bag) error for validation',
            'Increase n_estimators until error plateaus',
            'Feature importance can guide feature selection',
            'Use class_weight for imbalanced data',
            'n_jobs=-1 for parallel training'
        ],

        exampleDataset: `feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
3.4,5.6,7.8,0
4.5,6.7,8.9,1`,

        exampleOutput: `{
  "predictions": [0, 1, 0, 1],
  "probabilities": [[0.87, 0.13], [0.22, 0.78], ...],
  "oob_score": 0.89,
  "feature_importance": [0.35, 0.42, 0.23]
}`,

        references: [
            { title: 'Original Random Forest Paper', url: 'https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf' },
            { title: 'Understanding Random Forests', url: 'https://arxiv.org/abs/1407.7502' },
            { title: 'Scikit-learn Random Forest', url: 'https://scikit-learn.org/stable/modules/ensemble.html#random-forests' }
        ]
    },

    'knn': {
        id: 'knn',
        name: 'KNN',
        fullName: 'K-Nearest Neighbors',
        category: 'Classification',

        introduction: `K-Nearest Neighbors (KNN) is one of the simplest machine learning algorithms. It's a non-parametric, instance-based learning method that stores all training examples and classifies new data points based on the majority class of their k closest neighbors.

KNN is a "lazy learner" - it doesn't build an explicit model during training. Instead, all computation happens during prediction, making it simple but potentially slow for large datasets.`,

        mathematicalBackground: [
            {
                title: 'Euclidean Distance',
                content: 'Most common distance metric for continuous features.',
                formula: 'd(x,y) = √(Σᵢ(xᵢ - yᵢ)²)'
            },
            {
                title: 'Manhattan Distance',
                content: 'Sum of absolute differences, good for grid-like paths.',
                formula: 'd(x,y) = Σᵢ|xᵢ - yᵢ|'
            },
            {
                title: 'Weighted Voting',
                content: 'Closer neighbors can have more influence using distance weighting.',
                formula: 'weight(i) = 1 / d(x, xᵢ)'
            }
        ],

        pseudocode: `Algorithm: K-Nearest Neighbors
----------------------------------------
TRAINING:
1. Store all training examples (X, y)

PREDICTION:
For each test sample x:
1. Compute distance to all training samples
2. Select K nearest neighbors
3. For classification:
   - Return majority class among K neighbors
   - Optionally weight by distance
4. For regression:
   - Return mean (or weighted mean) of K values

CHOOSING K:
1. Use cross-validation
2. Odd K avoids ties for binary classification
3. K = √n is a common heuristic`,

        complexity: {
            time: 'O(1) training, O(n × d) per prediction',
            space: 'O(n × d) for storing all training data'
        },

        whenToUse: [
            'Simple baseline needed quickly',
            'Decision boundary is irregular',
            'Small to medium dataset',
            'Multi-class classification',
            'Data is relatively low-dimensional'
        ],

        whenNotToUse: [
            'Large datasets (slow predictions)',
            'High-dimensional data (curse of dimensionality)',
            'Imbalanced classes',
            'Irrelevant features present',
            'Real-time predictions needed'
        ],

        tips: [
            'Always normalize/standardize features',
            'Use cross-validation to find optimal K',
            'Try different distance metrics',
            'Use KD-trees or Ball-trees for faster lookup',
            'Consider weighted KNN for better accuracy'
        ],

        exampleDataset: `x1,x2,class
1.0,1.0,A
1.2,0.8,A
5.0,5.0,B
5.2,4.8,B`,

        exampleOutput: `{
  "prediction": "A",
  "neighbors": [
    {"point": [1.0, 1.0], "class": "A", "distance": 0.5},
    {"point": [1.2, 0.8], "class": "A", "distance": 0.6},
    {"point": [5.0, 5.0], "class": "B", "distance": 4.2}
  ],
  "k": 3,
  "accuracy": 95.5
}`,

        references: [
            { title: 'KNN Tutorial', url: 'https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html' },
            { title: 'Curse of Dimensionality', url: 'https://stats.stackexchange.com/questions/99171/curse-of-dimensionality' },
            { title: 'Scikit-learn KNN', url: 'https://scikit-learn.org/stable/modules/neighbors.html' }
        ]
    },

    'hierarchical-clustering': {
        id: 'hierarchical-clustering',
        name: 'Hierarchical Clustering',
        fullName: 'Agglomerative Hierarchical Clustering',
        category: 'Clustering',

        introduction: `Hierarchical Clustering builds a tree of clusters (dendrogram) by iteratively merging or splitting groups. The agglomerative (bottom-up) approach starts with each point as its own cluster and merges the closest pairs.

Unlike K-Means, it doesn't require specifying the number of clusters upfront. You can cut the dendrogram at different heights to get different numbers of clusters, making it flexible for exploratory analysis.`,

        mathematicalBackground: [
            {
                title: 'Single Linkage',
                content: 'Distance between clusters is the minimum distance between any points.',
                formula: 'd(A,B) = min{d(a,b) : a∈A, b∈B}'
            },
            {
                title: 'Complete Linkage',
                content: 'Distance is the maximum distance between any points.',
                formula: 'd(A,B) = max{d(a,b) : a∈A, b∈B}'
            },
            {
                title: 'Ward Linkage',
                content: 'Minimizes the total within-cluster variance.',
                formula: 'Δ(A,B) = |A||B|/(|A|+|B|) × ||c_A - c_B||²'
            }
        ],

        pseudocode: `Algorithm: Agglomerative Clustering
----------------------------------------
1. Initialize: each point is its own cluster
2. Compute distance matrix D[i,j] for all pairs
3. Repeat until one cluster remains:
   a. Find two closest clusters (i,j)
   b. Merge i and j into new cluster
   c. Update distances using linkage method
   d. Record merge in dendrogram
4. Cut dendrogram at desired level
5. Return cluster assignments`,

        complexity: {
            time: 'O(n³) naive, O(n² log n) with efficient data structures',
            space: 'O(n²) for distance matrix'
        },

        whenToUse: [
            'Exploring hierarchical structure in data',
            'Unknown number of clusters',
            'Need dendrogram visualization',
            'Small to medium datasets',
            'Taxonomies or biology applications'
        ],

        whenNotToUse: [
            'Large datasets (memory issues)',
            'Need fast clustering',
            'Spherical cluster assumption is OK (use K-Means)',
            'Clusters have very different densities',
            'Real-time clustering needed'
        ],

        tips: [
            'Use Ward linkage for compact clusters',
            'Single linkage can create chaining effect',
            'Cut dendrogram where gap is largest',
            'Visualize dendrogram to understand structure',
            'Consider HDBSCAN for large datasets'
        ],

        exampleDataset: `x,y
1.0,1.0
1.2,0.8
5.0,5.0
5.2,4.8
10.0,10.0`,

        exampleOutput: `{
  "labels": [0, 0, 1, 1, 2],
  "n_clusters": 3,
  "dendrogram": {
    "merges": [[0,1], [2,3], [4,5], [6,7]],
    "distances": [0.28, 0.28, 5.66, 7.07]
  }
}`,

        references: [
            { title: 'Hierarchical Clustering Overview', url: 'https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-clustering-1.html' },
            { title: 'Linkage Methods Comparison', url: 'https://stats.stackexchange.com/questions/195446/choosing-the-right-linkage-method-for-hierarchical-clustering' },
            { title: 'Scikit-learn Hierarchical', url: 'https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering' }
        ]
    },

    'quick-sort': {
        id: 'quick-sort',
        name: 'Quick Sort',
        fullName: 'Quick Sort Algorithm',
        category: 'Sorting',

        introduction: `Quick Sort is one of the most efficient and widely used sorting algorithms. It uses a divide-and-conquer strategy: select a pivot element, partition the array around it (smaller elements to the left, larger to the right), and recursively sort the partitions.

Developed by Tony Hoare in 1959, Quick Sort is the default sorting algorithm in many programming languages due to its excellent average-case performance and cache efficiency.`,

        mathematicalBackground: [
            {
                title: 'Average Case',
                content: 'When partitions are balanced, we get optimal performance.',
                formula: 'T(n) = 2T(n/2) + O(n) = O(n log n)'
            },
            {
                title: 'Worst Case',
                content: 'When pivot is always min or max (already sorted data).',
                formula: 'T(n) = T(n-1) + O(n) = O(n²)'
            },
            {
                title: 'Space Complexity',
                content: 'In-place but uses stack space for recursion.',
                formula: 'O(log n) average, O(n) worst case'
            }
        ],

        pseudocode: `Algorithm: Quick Sort
----------------------------------------
QUICKSORT(A, low, high):
1. If low < high:
   a. pivot_index = PARTITION(A, low, high)
   b. QUICKSORT(A, low, pivot_index - 1)
   c. QUICKSORT(A, pivot_index + 1, high)

PARTITION(A, low, high):
1. pivot = A[high]  // or use median-of-three
2. i = low - 1
3. For j = low to high - 1:
   a. If A[j] <= pivot:
      i = i + 1
      Swap A[i] and A[j]
4. Swap A[i+1] and A[high]
5. Return i + 1`,

        complexity: {
            time: 'O(n log n) average, O(n²) worst case',
            space: 'O(log n) for recursion stack'
        },

        whenToUse: [
            'General-purpose sorting',
            'Arrays (cache-friendly)',
            'Average case performance matters',
            'Memory is limited (in-place)',
            'Data is randomly ordered'
        ],

        whenNotToUse: [
            'Data is already nearly sorted',
            'Stability is required',
            'Worst case must be avoided',
            'Sorting linked lists',
            'Parallel sorting needed (use merge sort)'
        ],

        tips: [
            'Use median-of-three pivot selection',
            'Switch to insertion sort for small subarrays',
            'Use 3-way partitioning for many duplicates',
            'Tail recursion optimization saves stack space',
            'Randomize pivot to avoid worst case'
        ],

        exampleDataset: `array: [64, 34, 25, 12, 22, 11, 90]`,

        exampleOutput: `{
  "sorted": [11, 12, 22, 25, 34, 64, 90],
  "comparisons": 16,
  "swaps": 9,
  "pivots_used": [90, 34, 12, 25]
}`,

        references: [
            { title: 'Quick Sort Analysis', url: 'https://www.cs.cornell.edu/courses/cs2110/2016sp/recitations/qs.pdf' },
            { title: 'Pivot Selection Strategies', url: 'https://www.geeksforgeeks.org/quick-sort/' },
            { title: 'Why Quick Sort is Fast', url: 'https://cs.stackexchange.com/questions/3/why-is-quicksort-better-than-other-sorting-algorithms-in-practice' }
        ]
    },

    'merge-sort': {
        id: 'merge-sort',
        name: 'Merge Sort',
        fullName: 'Merge Sort Algorithm',
        category: 'Sorting',

        introduction: `Merge Sort is a stable, divide-and-conquer sorting algorithm that guarantees O(n log n) time complexity in all cases. It divides the array into halves, recursively sorts them, and merges the sorted halves.

Invented by John von Neumann in 1945, Merge Sort is particularly useful for sorting linked lists (where it excels over Quick Sort) and for external sorting of data too large to fit in memory.`,

        mathematicalBackground: [
            {
                title: 'Recurrence Relation',
                content: 'The time complexity is the same in all cases.',
                formula: 'T(n) = 2T(n/2) + O(n) = O(n log n)'
            },
            {
                title: 'Space Complexity',
                content: 'Requires additional space for merging.',
                formula: 'O(n) auxiliary space'
            },
            {
                title: 'Merge Operation',
                content: 'Merging two sorted arrays of size n/2 takes linear time.',
                formula: 'O(n) for each merge level'
            }
        ],

        pseudocode: `Algorithm: Merge Sort
----------------------------------------
MERGESORT(A):
1. If length(A) <= 1: return A
2. mid = length(A) / 2
3. left = MERGESORT(A[0..mid])
4. right = MERGESORT(A[mid..end])
5. Return MERGE(left, right)

MERGE(left, right):
1. result = []
2. i = 0, j = 0
3. While i < len(left) and j < len(right):
   a. If left[i] <= right[j]:
      Append left[i] to result, i++
   b. Else:
      Append right[j] to result, j++
4. Append remaining elements
5. Return result`,

        complexity: {
            time: 'O(n log n) in all cases',
            space: 'O(n) auxiliary space'
        },

        whenToUse: [
            'Guaranteed O(n log n) performance needed',
            'Sorting linked lists',
            'Stability is required',
            'External sorting (disk-based)',
            'Parallel processing available'
        ],

        whenNotToUse: [
            'Memory is very limited',
            'Sorting small arrays (use insertion sort)',
            'In-place sorting required',
            'Random access is expensive',
            'Simple implementation preferred'
        ],

        tips: [
            'Use for external sorting with disk I/O',
            'Natural merge sort optimizes for partially sorted data',
            'Bottom-up merge sort avoids recursion overhead',
            'Parallelize easily by sorting halves independently',
            'Combine with insertion sort for small subarrays'
        ],

        exampleDataset: `array: [38, 27, 43, 3, 9, 82, 10]`,

        exampleOutput: `{
  "sorted": [3, 9, 10, 27, 38, 43, 82],
  "comparisons": 13,
  "merge_operations": 6,
  "recursion_depth": 3
}`,

        references: [
            { title: 'Merge Sort Visualization', url: 'https://visualgo.net/en/sorting' },
            { title: 'External Merge Sort', url: 'https://en.wikipedia.org/wiki/External_sorting' },
            { title: 'Comparison with Quick Sort', url: 'https://stackoverflow.com/questions/680541/quick-sort-vs-merge-sort' }
        ]
    },

    'bubble-sort': {
        id: 'bubble-sort',
        name: 'Bubble Sort',
        fullName: 'Bubble Sort Algorithm',
        category: 'Sorting',

        introduction: `Bubble Sort is the simplest sorting algorithm. It repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. The pass is repeated until no swaps are needed.

While inefficient for large datasets (O(n²)), Bubble Sort is valuable for educational purposes and can be optimal for nearly sorted data when using an early termination flag.`,

        mathematicalBackground: [
            {
                title: 'Comparisons',
                content: 'Total comparisons in worst case.',
                formula: 'n(n-1)/2 = O(n²)'
            },
            {
                title: 'Best Case',
                content: 'Already sorted data with early termination.',
                formula: 'O(n) with optimization'
            },
            {
                title: 'Inversions',
                content: 'Bubble sort swaps once per inversion.',
                formula: 'Swaps = number of inversions'
            }
        ],

        pseudocode: `Algorithm: Bubble Sort
----------------------------------------
BUBBLESORT(A):
1. n = length(A)
2. For i = 0 to n-1:
   a. swapped = false
   b. For j = 0 to n-i-2:
      If A[j] > A[j+1]:
         Swap A[j] and A[j+1]
         swapped = true
   c. If not swapped:
      Break  // Array is sorted
3. Return A

// Each pass "bubbles" the largest
// unsorted element to its position`,

        complexity: {
            time: 'O(n²) average and worst, O(n) best (sorted)',
            space: 'O(1) in-place'
        },

        whenToUse: [
            'Teaching sorting concepts',
            'Very small datasets (n < 20)',
            'Data is nearly sorted',
            'Simplicity over performance',
            'Memory is extremely limited'
        ],

        whenNotToUse: [
            'Any production application',
            'Datasets larger than a few dozen elements',
            'Performance matters at all',
            'Random or reverse-sorted data',
            'Better alternatives exist (always)'
        ],

        tips: [
            'Always use the swapped flag for early termination',
            'Cocktail sort is a bidirectional variant',
            'Use only for educational purposes',
            'Consider insertion sort instead (also O(n²) but faster)',
            'Good for detecting already sorted arrays'
        ],

        exampleDataset: `array: [64, 34, 25, 12, 22, 11, 90]`,

        exampleOutput: `{
  "sorted": [11, 12, 22, 25, 34, 64, 90],
  "comparisons": 21,
  "swaps": 16,
  "passes": 6
}`,

        references: [
            { title: 'Bubble Sort Analysis', url: 'https://www.cs.umd.edu/~meesh/351/mount/lectures/lect14-sorting1.pdf' },
            { title: 'Why Bubble Sort is Bad', url: 'https://cs.stackexchange.com/questions/62588/why-is-bubble-sort-so-slow' },
            { title: 'Sorting Algorithms Comparison', url: 'https://www.toptal.com/developers/sorting-algorithms' }
        ]
    },

    'binary-search': {
        id: 'binary-search',
        name: 'Binary Search',
        fullName: 'Binary Search Algorithm',
        category: 'Searching',

        introduction: `Binary Search is one of the most efficient algorithms for finding an element in a sorted array. By repeatedly dividing the search interval in half, it achieves O(log n) time complexity.

The algorithm is fundamental in computer science and forms the basis for many advanced data structures like binary search trees, B-trees, and is used extensively in system utilities like git bisect.`,

        mathematicalBackground: [
            {
                title: 'Time Complexity',
                content: 'Each comparison halves the search space.',
                formula: 'T(n) = T(n/2) + O(1) = O(log n)'
            },
            {
                title: 'Maximum Comparisons',
                content: 'At most log₂(n) + 1 comparisons needed.',
                formula: '⌊log₂(n)⌋ + 1'
            },
            {
                title: 'Mid Calculation',
                content: 'Avoid overflow with proper formula.',
                formula: 'mid = low + (high - low) / 2'
            }
        ],

        pseudocode: `Algorithm: Binary Search
----------------------------------------
BINARY_SEARCH(A, target):
1. low = 0, high = length(A) - 1
2. While low <= high:
   a. mid = low + (high - low) / 2
   b. If A[mid] == target:
      Return mid
   c. Else if A[mid] < target:
      low = mid + 1
   d. Else:
      high = mid - 1
3. Return -1  // Not found

VARIATIONS:
- Find first occurrence
- Find last occurrence
- Find insertion position
- Search in rotated array`,

        complexity: {
            time: 'O(log n)',
            space: 'O(1) iterative, O(log n) recursive'
        },

        whenToUse: [
            'Searching in sorted arrays',
            'Database index lookups',
            'Finding boundaries (lower/upper bound)',
            'Optimization problems (binary search on answer)',
            'Version control (git bisect)'
        ],

        whenNotToUse: [
            'Unsorted data',
            'Small arrays (linear search is simpler)',
            'Linked lists (no random access)',
            'Frequently changing data',
            'When insertion/deletion is common'
        ],

        tips: [
            'Use low + (high - low) / 2 to avoid overflow',
            'Be careful with off-by-one errors',
            'Consider lower_bound/upper_bound variants',
            'Binary search can find insertion points',
            'Works on any monotonic function'
        ],

        exampleDataset: `sorted_array: [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
target: 23`,

        exampleOutput: `{
  "found": true,
  "index": 5,
  "comparisons": 3,
  "search_path": [
    {"mid": 4, "value": 16, "action": "go right"},
    {"mid": 7, "value": 56, "action": "go left"},
    {"mid": 5, "value": 23, "action": "found"}
  ]
}`,

        references: [
            { title: 'Binary Search Pitfalls', url: 'https://ai.googleblog.com/2006/06/extra-extra-read-all-about-it-nearly.html' },
            { title: 'Binary Search Variations', url: 'https://www.topcoder.com/thrive/articles/Binary%20Search' },
            { title: 'Git Bisect', url: 'https://git-scm.com/docs/git-bisect' }
        ]
    },

    'dijkstra': {
        id: 'dijkstra',
        name: 'Dijkstra',
        fullName: "Dijkstra's Shortest Path Algorithm",
        category: 'Graph',

        introduction: `Dijkstra's algorithm finds the shortest paths from a source vertex to all other vertices in a weighted graph with non-negative edge weights. It's a greedy algorithm that always expands the vertex with the smallest known distance.

Conceived by Edsger W. Dijkstra in 1956 and published in 1959, it's one of the most famous algorithms in computer science, used in GPS navigation, network routing, and countless other applications.`,

        mathematicalBackground: [
            {
                title: 'Relaxation',
                content: 'Update distance if a shorter path is found.',
                formula: 'd[v] = min(d[v], d[u] + w(u,v))'
            },
            {
                title: 'Greedy Choice',
                content: 'Always process the vertex with minimum distance.',
                formula: 'u = argmin{d[v] : v ∈ unvisited}'
            },
            {
                title: 'Optimality',
                content: 'Dijkstra finds optimal paths when all weights are non-negative.',
                formula: 'w(u,v) ≥ 0 for all edges'
            }
        ],

        pseudocode: `Algorithm: Dijkstra's Shortest Path
----------------------------------------
DIJKSTRA(G, source):
1. Initialize:
   d[source] = 0
   d[v] = ∞ for all other v
   prev[v] = null for all v
   Q = priority queue with all vertices
2. While Q is not empty:
   a. u = extract_min(Q)
   b. For each neighbor v of u:
      alt = d[u] + weight(u, v)
      If alt < d[v]:
         d[v] = alt
         prev[v] = u
         decrease_key(Q, v, alt)
3. Return d[], prev[]

PATH RECONSTRUCTION:
Follow prev[] pointers from target to source`,

        complexity: {
            time: 'O((V + E) log V) with binary heap',
            space: 'O(V) for distances and predecessors'
        },

        whenToUse: [
            'Shortest path in weighted graphs',
            'All edge weights are non-negative',
            'GPS and navigation systems',
            'Network routing protocols',
            'Game AI pathfinding (with A* heuristic)'
        ],

        whenNotToUse: [
            'Negative edge weights (use Bellman-Ford)',
            'All-pairs shortest path (use Floyd-Warshall)',
            'Unweighted graphs (use BFS)',
            'Very dense graphs (matrix-based may be faster)',
            'Single-pair with good heuristic (use A*)'
        ],

        tips: [
            'Use binary heap for efficient implementation',
            'Consider A* if you have a good heuristic',
            'Fibonacci heap gives O(E + V log V) but complex',
            'Bidirectional Dijkstra can speed up point-to-point',
            'Pre-compute for frequently queried graphs'
        ],

        exampleDataset: `graph:
  A --1--> B
  A --4--> C
  B --2--> C
  B --5--> D
  C --1--> D`,

        exampleOutput: `{
  "source": "A",
  "distances": {"A": 0, "B": 1, "C": 3, "D": 4},
  "predecessors": {"B": "A", "C": "B", "D": "C"},
  "path_to_D": ["A", "B", "C", "D"],
  "vertices_processed": 4
}`,

        references: [
            { title: 'Original Dijkstra Paper', url: 'https://dl.acm.org/doi/10.1007/BF01386390' },
            { title: 'Dijkstra vs A*', url: 'https://www.redblobgames.com/pathfinding/a-star/introduction.html' },
            { title: 'Priority Queue Implementations', url: 'https://cs.stackexchange.com/questions/10038/dijkstras-algorithm-with-fibonacci-heap' }
        ]
    },

    'bfs-dfs': {
        id: 'bfs-dfs',
        name: 'BFS/DFS',
        fullName: 'Breadth-First Search / Depth-First Search',
        category: 'Graph',

        introduction: `BFS and DFS are fundamental graph traversal algorithms. BFS explores level by level (using a queue), while DFS explores as deep as possible before backtracking (using a stack or recursion).

BFS finds shortest paths in unweighted graphs and is ideal for finding nearby nodes first. DFS is memory-efficient for deep graphs and is fundamental for topological sorting, cycle detection, and strongly connected components.`,

        mathematicalBackground: [
            {
                title: 'BFS Shortest Path',
                content: 'In unweighted graphs, BFS finds the shortest path.',
                formula: 'd(s,v) = minimum edges from s to v'
            },
            {
                title: 'DFS Discovery/Finish Times',
                content: 'DFS assigns timestamps useful for cycle detection.',
                formula: 'discovery[v] < discovery[u] < finish[u] < finish[v] (ancestor)'
            },
            {
                title: 'Space Complexity',
                content: 'BFS needs O(branching^depth), DFS needs O(depth).',
                formula: 'BFS: O(b^d), DFS: O(d × b)'
            }
        ],

        pseudocode: `Algorithm: BFS
----------------------------------------
BFS(G, source):
1. Mark source as visited, enqueue source
2. While queue not empty:
   a. u = dequeue()
   b. For each neighbor v of u:
      If v not visited:
         Mark v visited
         parent[v] = u
         enqueue(v)

Algorithm: DFS
----------------------------------------
DFS(G, source):
1. Mark source as visited
2. For each neighbor v of source:
   If v not visited:
      parent[v] = source
      DFS(G, v)

// Or iterative with explicit stack`,

        complexity: {
            time: 'O(V + E) for both BFS and DFS',
            space: 'O(V) for visited set and queue/stack'
        },

        whenToUse: [
            'BFS: Shortest path in unweighted graphs',
            'BFS: Level-order traversal',
            'DFS: Cycle detection',
            'DFS: Topological sorting',
            'Both: Connected components'
        ],

        whenNotToUse: [
            'BFS: Very deep graphs (memory intensive)',
            'DFS: Need shortest path (doesn\'t guarantee)',
            'Weighted graphs (use Dijkstra)',
            'Infinite graphs without bounds',
            'Need optimal paths always'
        ],

        tips: [
            'BFS is level-order, DFS is depth-first',
            'Use BFS for shortest path in unweighted graphs',
            'DFS can be recursive or iterative',
            'Mark nodes before adding to queue (BFS)',
            'DFS finish times give topological order'
        ],

        exampleDataset: `graph:
  1 -- 2 -- 5
  |    |
  3 -- 4`,

        exampleOutput: `{
  "bfs_order": [1, 2, 3, 4, 5],
  "bfs_distances": {1: 0, 2: 1, 3: 1, 4: 2, 5: 2},
  "dfs_order": [1, 2, 4, 3, 5],
  "dfs_discovery": {1: 1, 2: 2, 4: 3, 3: 4, 5: 6},
  "dfs_finish": {3: 5, 4: 7, 5: 8, 2: 9, 1: 10}
}`,

        references: [
            { title: 'Graph Traversal Overview', url: 'https://www.cs.cornell.edu/courses/cs2112/2012sp/lectures/lec24/lec24-12sp.html' },
            { title: 'BFS vs DFS Comparison', url: 'https://www.geeksforgeeks.org/difference-between-bfs-and-dfs/' },
            { title: 'Applications of DFS', url: 'https://cp-algorithms.com/graph/depth-first-search.html' }
        ]
    },

    // === NEW SORTING ALGORITHMS ===
    'heap-sort': {
        id: 'heap-sort',
        name: 'Heap Sort',
        fullName: 'Heap Sort Algorithm',
        category: 'Sorting',
        introduction: `Heap Sort uses a binary heap data structure to sort elements. It builds a max-heap from the input, then repeatedly extracts the maximum element to build the sorted array.`,
        mathematicalBackground: [
            { title: 'Heap Property', content: 'Parent ≥ children (max-heap) or Parent ≤ children (min-heap).', formula: 'parent(i) = (i-1)/2' },
            { title: 'Time Complexity', content: 'Building heap is O(n), extraction is O(n log n).', formula: 'T(n) = O(n log n)' }
        ],
        pseudocode: `HEAPSORT(A):\n1. BUILD_MAX_HEAP(A)\n2. for i = n-1 to 1:\n   swap A[0] with A[i]\n   heapify(A, 0, i)`,
        complexity: { time: 'O(n log n) all cases', space: 'O(1) in-place' },
        whenToUse: ['Guaranteed O(n log n) needed', 'In-place sorting required', 'Priority queue operations'],
        whenNotToUse: ['Stability required', 'Cache efficiency important', 'Small datasets'],
        tips: ['Use for priority queues', 'Good when worst case must be avoided'],
        exampleDataset: `[64, 34, 25, 12, 22, 11, 90]`,
        exampleOutput: `{"sorted": [11, 12, 22, 25, 34, 64, 90]}`,
        references: [{ title: 'Heap Sort', url: 'https://www.geeksforgeeks.org/heap-sort/' }]
    },

    'insertion-sort': {
        id: 'insertion-sort',
        name: 'Insertion Sort',
        fullName: 'Insertion Sort Algorithm',
        category: 'Sorting',
        introduction: `Insertion Sort builds the sorted array one element at a time by inserting each element into its correct position among the already sorted elements.`,
        mathematicalBackground: [
            { title: 'Comparisons', content: 'Best case O(n), worst case O(n²).', formula: 'Worst: n(n-1)/2' }
        ],
        pseudocode: `INSERTION_SORT(A):\nfor i = 1 to n-1:\n  key = A[i]\n  j = i - 1\n  while j >= 0 and A[j] > key:\n    A[j+1] = A[j]\n    j--\n  A[j+1] = key`,
        complexity: { time: 'O(n²) worst, O(n) best', space: 'O(1)' },
        whenToUse: ['Small datasets', 'Nearly sorted data', 'Online sorting'],
        whenNotToUse: ['Large datasets', 'Random data'],
        tips: ['Excellent for nearly sorted data', 'Often used as base case in hybrid sorts'],
        exampleDataset: `[12, 11, 13, 5, 6]`,
        exampleOutput: `{"sorted": [5, 6, 11, 12, 13]}`,
        references: [{ title: 'Insertion Sort', url: 'https://www.geeksforgeeks.org/insertion-sort/' }]
    },

    'linear-search': {
        id: 'linear-search',
        name: 'Linear Search',
        fullName: 'Linear Search Algorithm',
        category: 'Searching',
        introduction: `Linear Search sequentially checks each element of a list until a match is found or the list is exhausted. It works on unsorted data.`,
        mathematicalBackground: [
            { title: 'Time Complexity', content: 'Checks each element once.', formula: 'T(n) = O(n)' }
        ],
        pseudocode: `LINEAR_SEARCH(A, target):\nfor i = 0 to n-1:\n  if A[i] == target:\n    return i\nreturn -1`,
        complexity: { time: 'O(n)', space: 'O(1)' },
        whenToUse: ['Unsorted data', 'Small lists', 'Single search'],
        whenNotToUse: ['Sorted data (use binary search)', 'Repeated searches'],
        tips: ['Simplest search algorithm', 'No preprocessing needed'],
        exampleDataset: `array: [10, 20, 30, 40]\ntarget: 30`,
        exampleOutput: `{"found": true, "index": 2}`,
        references: [{ title: 'Linear Search', url: 'https://www.geeksforgeeks.org/linear-search/' }]
    },

    // === NEW GRAPH ALGORITHMS ===
    'a-star': {
        id: 'a-star',
        name: 'A*',
        fullName: 'A* Search Algorithm',
        category: 'Graph',
        introduction: `A* is a best-first search algorithm that finds the shortest path using a heuristic. It combines Dijkstra's completeness with greedy best-first search efficiency.`,
        mathematicalBackground: [
            { title: 'Evaluation Function', content: 'f(n) = g(n) + h(n)', formula: 'f(n) = g(n) + h(n)' },
            { title: 'Admissibility', content: 'h(n) must never overestimate the actual cost.', formula: 'h(n) ≤ h*(n)' }
        ],
        pseudocode: `A_STAR(start, goal):\nopen = PriorityQueue([start])\nwhile open not empty:\n  current = open.pop(lowest f)\n  if current == goal: return path\n  for neighbor in current.neighbors:\n    g = g[current] + cost\n    if g < g[neighbor]:\n      update g, f, parent\n      add to open`,
        complexity: { time: 'O(b^d) worst case', space: 'O(b^d)' },
        whenToUse: ['Game pathfinding', 'Robot navigation', 'Good heuristic available'],
        whenNotToUse: ['No good heuristic', 'Memory constrained'],
        tips: ['Use Manhattan distance for grids', 'Euclidean for open spaces'],
        exampleDataset: `grid with obstacles, start (0,0), goal (9,9)`,
        exampleOutput: `{"path": [[0,0], [1,1], ..., [9,9]], "cost": 14}`,
        references: [{ title: 'A* Algorithm', url: 'https://www.redblobgames.com/pathfinding/a-star/introduction.html' }]
    },

    'bellman-ford': {
        id: 'bellman-ford',
        name: 'Bellman-Ford',
        fullName: 'Bellman-Ford Algorithm',
        category: 'Graph',
        introduction: `Bellman-Ford computes shortest paths from a source vertex, handling negative edge weights. It can detect negative cycles.`,
        mathematicalBackground: [
            { title: 'Relaxation', content: 'Update distance if shorter path found.', formula: 'd[v] = min(d[v], d[u] + w(u,v))' }
        ],
        pseudocode: `BELLMAN_FORD(G, s):\nfor v in V: d[v] = ∞\nd[s] = 0\nfor i = 1 to |V|-1:\n  for each edge (u,v,w):\n    if d[u] + w < d[v]:\n      d[v] = d[u] + w`,
        complexity: { time: 'O(V × E)', space: 'O(V)' },
        whenToUse: ['Negative edge weights', 'Detect negative cycles'],
        whenNotToUse: ['Non-negative weights (use Dijkstra)'],
        tips: ['One more iteration detects negative cycles'],
        exampleDataset: `graph with negative weights`,
        exampleOutput: `{"distances": {"A": 0, "B": -1, "C": 2}}`,
        references: [{ title: 'Bellman-Ford', url: 'https://www.geeksforgeeks.org/bellman-ford-algorithm-dp-23/' }]
    },

    'floyd-warshall': {
        id: 'floyd-warshall',
        name: 'Floyd-Warshall',
        fullName: 'Floyd-Warshall Algorithm',
        category: 'Graph',
        introduction: `Floyd-Warshall finds shortest paths between all pairs of vertices using dynamic programming.`,
        mathematicalBackground: [
            { title: 'DP Recurrence', content: 'Consider all intermediate vertices.', formula: 'dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])' }
        ],
        pseudocode: `FLOYD_WARSHALL(W):\ndist = W\nfor k = 1 to n:\n  for i = 1 to n:\n    for j = 1 to n:\n      dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`,
        complexity: { time: 'O(V³)', space: 'O(V²)' },
        whenToUse: ['All-pairs shortest path', 'Dense graphs'],
        whenNotToUse: ['Single source (use Dijkstra)', 'Sparse graphs'],
        tips: ['Good for small dense graphs'],
        exampleDataset: `4x4 adjacency matrix`,
        exampleOutput: `{"allPairsDistances": [[0,3,8,inf],[inf,0,inf,1],...]}`,
        references: [{ title: 'Floyd-Warshall', url: 'https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/' }]
    },

    'pagerank': {
        id: 'pagerank',
        name: 'PageRank',
        fullName: 'PageRank Algorithm',
        category: 'Graph',
        introduction: `PageRank ranks nodes by importance based on incoming links. A page is important if linked by other important pages.`,
        mathematicalBackground: [
            { title: 'PageRank Formula', content: 'Iterative computation with damping.', formula: 'PR(A) = (1-d)/N + d × Σ PR(Ti)/C(Ti)' }
        ],
        pseudocode: `PAGERANK(G, d=0.85):\nN = |V|\nfor v in V: PR[v] = 1/N\nrepeat until convergence:\n  for v in V:\n    PR[v] = (1-d)/N + d * sum(PR[u]/out[u] for u in inlinks[v])`,
        complexity: { time: 'O(E × iterations)', space: 'O(V)' },
        whenToUse: ['Web page ranking', 'Social network analysis', 'Citation ranking'],
        whenNotToUse: ['Undirected graphs', 'Real-time requirements'],
        tips: ['d=0.85 is standard', '100 iterations usually sufficient'],
        exampleDataset: `web graph with links`,
        exampleOutput: `{"ranks": {"A": 0.28, "B": 0.24, "C": 0.48}}`,
        references: [{ title: 'The PageRank Paper', url: 'http://ilpubs.stanford.edu:8090/422/' }]
    },

    // === DEEP LEARNING ===
    'neural-network': {
        id: 'neural-network',
        name: 'Neural Network',
        fullName: 'Artificial Neural Network',
        category: 'Deep Learning',
        introduction: `Neural networks are computing systems inspired by biological brains. They learn patterns through layers of interconnected nodes.`,
        mathematicalBackground: [
            { title: 'Neuron Output', content: 'Weighted sum plus activation.', formula: 'y = σ(Σ wᵢxᵢ + b)' },
            { title: 'Backpropagation', content: 'Gradient descent to minimize loss.', formula: '∂L/∂w using chain rule' }
        ],
        pseudocode: `FORWARD(x):\nfor layer in layers:\n  x = activation(layer.weights @ x + layer.bias)\nreturn x\n\nBACKWARD(loss):\nfor layer in reversed(layers):\n  gradient = compute_gradient(loss)\n  layer.weights -= lr * gradient`,
        complexity: { time: 'O(n×d×layers) per sample', space: 'O(parameters)' },
        whenToUse: ['Complex pattern recognition', 'Large datasets', 'Non-linear relationships'],
        whenNotToUse: ['Small datasets', 'Interpretability required', 'Limited compute'],
        tips: ['Start simple, add complexity', 'Use batch normalization', 'Monitor for overfitting'],
        exampleDataset: `images or tabular data with features`,
        exampleOutput: `{"prediction": 0.87, "class": "cat"}`,
        references: [{ title: 'Neural Networks', url: 'http://neuralnetworksanddeeplearning.com/' }]
    },

    'cnn': {
        id: 'cnn',
        name: 'CNN',
        fullName: 'Convolutional Neural Network',
        category: 'Deep Learning',
        introduction: `CNNs are specialized for image processing using convolutional layers that automatically learn spatial feature hierarchies.`,
        mathematicalBackground: [
            { title: 'Convolution', content: 'Filter slides over input.', formula: '(f * g)[i,j] = ΣΣ f[m,n] × g[i-m,j-n]' }
        ],
        pseudocode: `CNN_FORWARD(image):\nx = image\nfor layer in conv_layers:\n  x = pool(relu(conv(x, layer.filters)))\nx = flatten(x)\nfor layer in fc_layers:\n  x = relu(layer.weights @ x + layer.bias)\nreturn softmax(x)`,
        complexity: { time: 'O(n×k²×c×filters)', space: 'O(parameters + activations)' },
        whenToUse: ['Image classification', 'Object detection', 'Computer vision'],
        whenNotToUse: ['Non-image data', 'No GPU available', 'Small datasets'],
        tips: ['Use pretrained models', 'Data augmentation helps', 'Use GPU'],
        exampleDataset: `28x28 grayscale images (MNIST)`,
        exampleOutput: `{"prediction": 7, "confidence": 0.98}`,
        references: [{ title: 'CNN Explainer', url: 'https://poloclub.github.io/cnn-explainer/' }]
    },

    'rnn-lstm': {
        id: 'rnn-lstm',
        name: 'RNN/LSTM',
        fullName: 'Recurrent Neural Network / LSTM',
        category: 'Deep Learning',
        introduction: `RNNs process sequences with hidden state memory. LSTMs add gates to handle long-term dependencies.`,
        mathematicalBackground: [
            { title: 'RNN Hidden State', content: 'State transitions through sequence.', formula: 'hₜ = σ(Wₕhₜ₋₁ + Wₓxₜ + b)' },
            { title: 'LSTM Gates', content: 'Forget, input, output gates control information flow.', formula: 'fₜ = σ(Wf·[hₜ₋₁,xₜ] + bf)' }
        ],
        pseudocode: `LSTM_FORWARD(sequence):\nh, c = init_states\noutputs = []\nfor x in sequence:\n  f = sigmoid(Wf @ [h, x])\n  i = sigmoid(Wi @ [h, x])\n  c = f*c + i*tanh(Wc @ [h, x])\n  o = sigmoid(Wo @ [h, x])\n  h = o * tanh(c)\n  outputs.append(h)\nreturn outputs`,
        complexity: { time: 'O(T×hidden²)', space: 'O(T×hidden)' },
        whenToUse: ['Sequential data', 'Time series', 'NLP tasks'],
        whenNotToUse: ['Parallelization needed', 'Very long sequences'],
        tips: ['Use LSTM over vanilla RNN', 'Consider Transformer for long sequences'],
        exampleDataset: `text sequence or time series`,
        exampleOutput: `{"next_word": "the", "probability": 0.45}`,
        references: [{ title: 'Understanding LSTM', url: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/' }]
    },

    'transformer': {
        id: 'transformer',
        name: 'Transformer',
        fullName: 'Transformer Architecture',
        category: 'Deep Learning',
        introduction: `Transformers use self-attention to process sequences in parallel. They are the foundation of GPT, BERT, and modern LLMs.`,
        mathematicalBackground: [
            { title: 'Self-Attention', content: 'Attention weights for all positions.', formula: 'Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V' },
            { title: 'Multi-Head', content: 'Multiple attention heads capture different patterns.', formula: 'MultiHead = Concat(head₁,...,headₙ)Wᴼ' }
        ],
        pseudocode: `TRANSFORMER_BLOCK(x):\nq, k, v = x @ Wq, x @ Wk, x @ Wv\nattention = softmax(q @ k.T / sqrt(d)) @ v\nx = LayerNorm(x + attention)\nx = LayerNorm(x + FFN(x))\nreturn x`,
        complexity: { time: 'O(n² × d)', space: 'O(n² + nd)' },
        whenToUse: ['NLP tasks', 'Long sequences', 'Pre-training possible'],
        whenNotToUse: ['Very long sequences (use linear attention)', 'Limited compute'],
        tips: ['Use pre-trained models', 'Fine-tune on your task'],
        exampleDataset: `text for language modeling`,
        exampleOutput: `{"generated": "The quick brown fox..."}`,
        references: [{ title: 'Attention Is All You Need', url: 'https://arxiv.org/abs/1706.03762' }]
    },

    // === STATISTICS ===
    'descriptive-stats': {
        id: 'descriptive-stats',
        name: 'Mean/Median/Mode',
        fullName: 'Descriptive Statistics',
        category: 'Statistics',
        introduction: `Measures of central tendency that summarize data: mean (average), median (middle value), mode (most frequent).`,
        mathematicalBackground: [
            { title: 'Mean', content: 'Sum divided by count.', formula: 'μ = Σxᵢ / n' },
            { title: 'Median', content: 'Middle value when sorted.', formula: 'x₍ₙ₊₁₎/₂' }
        ],
        pseudocode: `MEAN(data): return sum(data) / len(data)\nMEDIAN(data): return sorted(data)[len(data)//2]\nMODE(data): return most_frequent(data)`,
        complexity: { time: 'O(n) mean, O(n log n) median', space: 'O(1) or O(n)' },
        whenToUse: ['Data summarization', 'Understanding distributions'],
        whenNotToUse: ['Need full distribution'],
        tips: ['Mean sensitive to outliers', 'Use median for skewed data'],
        exampleDataset: `[1, 2, 2, 3, 4, 5, 100]`,
        exampleOutput: `{"mean": 16.7, "median": 3, "mode": 2}`,
        references: [{ title: 'Descriptive Statistics', url: 'https://www.khanacademy.org/math/statistics-probability' }]
    },

    'standard-deviation': {
        id: 'standard-deviation',
        name: 'Standard Deviation',
        fullName: 'Standard Deviation & Variance',
        category: 'Statistics',
        introduction: `Measures spread of data from the mean. Variance is squared deviation, std dev is its square root.`,
        mathematicalBackground: [
            { title: 'Variance', content: 'Average squared deviation.', formula: 'σ² = Σ(xᵢ - μ)² / n' },
            { title: 'Standard Deviation', content: 'Square root of variance.', formula: 'σ = √(σ²)' }
        ],
        pseudocode: `STD_DEV(data):\nmean = sum(data) / len(data)\nvariance = sum((x - mean)² for x in data) / len(data)\nreturn sqrt(variance)`,
        complexity: { time: 'O(n)', space: 'O(1)' },
        whenToUse: ['Measure variability', 'Risk analysis', 'Quality control'],
        whenNotToUse: ['Non-normal distributions'],
        tips: ['Use n-1 for sample std dev'],
        exampleDataset: `[2, 4, 4, 4, 5, 5, 7, 9]`,
        exampleOutput: `{"mean": 5, "variance": 4, "stdDev": 2}`,
        references: [{ title: 'Standard Deviation', url: 'https://www.mathsisfun.com/data/standard-deviation.html' }]
    },

    'correlation': {
        id: 'correlation',
        name: 'Correlation',
        fullName: 'Pearson Correlation Coefficient',
        category: 'Statistics',
        introduction: `Measures linear relationship between two variables. Ranges from -1 to +1.`,
        mathematicalBackground: [
            { title: 'Pearson Correlation', content: 'Covariance normalized by std devs.', formula: 'r = Σ(xᵢ-x̄)(yᵢ-ȳ) / √(Σ(xᵢ-x̄)²Σ(yᵢ-ȳ)²)' }
        ],
        pseudocode: `CORRELATION(x, y):\nmx, my = mean(x), mean(y)\ncov = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))\nreturn cov / (std(x) * std(y) * len(x))`,
        complexity: { time: 'O(n)', space: 'O(1)' },
        whenToUse: ['Feature selection', 'Relationship analysis'],
        whenNotToUse: ['Non-linear relationships'],
        tips: ['Correlation ≠ causation', 'Sensitive to outliers'],
        exampleDataset: `x: [1,2,3,4,5], y: [2,4,5,4,5]`,
        exampleOutput: `{"correlation": 0.83}`,
        references: [{ title: 'Correlation', url: 'https://www.statisticshowto.com/probability-and-statistics/correlation-coefficient-formula/' }]
    },

    'bayesian-inference': {
        id: 'bayesian-inference',
        name: 'Bayesian Inference',
        fullName: 'Bayesian Statistical Inference',
        category: 'Statistics',
        introduction: `Updates probability estimates with new evidence using Bayes' theorem.`,
        mathematicalBackground: [
            { title: 'Bayes Theorem', content: 'Posterior from prior and likelihood.', formula: 'P(H|E) = P(E|H)P(H) / P(E)' }
        ],
        pseudocode: `BAYES_UPDATE(prior, likelihood, evidence):\nposterior = likelihood * prior / evidence\nreturn posterior`,
        complexity: { time: 'Varies', space: 'O(hypotheses)' },
        whenToUse: ['A/B testing', 'Medical diagnosis', 'Incorporating prior knowledge'],
        whenNotToUse: ['Objective priors needed'],
        tips: ['Choose prior carefully', 'Conjugate priors simplify computation'],
        exampleDataset: `prior belief + observed data`,
        exampleOutput: `{"posterior": 0.75}`,
        references: [{ title: 'Bayesian Inference', url: 'https://seeing-theory.brown.edu/bayesian-inference/index.html' }]
    },

    'markov-chain': {
        id: 'markov-chain',
        name: 'Markov Chain',
        fullName: 'Markov Chain',
        category: 'Statistics',
        introduction: `A stochastic model where future state depends only on current state, not history.`,
        mathematicalBackground: [
            { title: 'Markov Property', content: 'Memoryless transitions.', formula: 'P(Xₙ₊₁|Xₙ,...,X₁) = P(Xₙ₊₁|Xₙ)' }
        ],
        pseudocode: `MARKOV_STEP(state, transition_matrix):\nprobs = transition_matrix[state]\nreturn random_choice(states, probs)`,
        complexity: { time: 'O(1) per step', space: 'O(states²)' },
        whenToUse: ['Text generation', 'Weather prediction', 'Random walk analysis'],
        whenNotToUse: ['Long-term dependencies matter'],
        tips: ['Compute stationary distribution for long-run behavior'],
        exampleDataset: `transition matrix between states`,
        exampleOutput: `{"stationary": [0.4, 0.6]}`,
        references: [{ title: 'Markov Chains', url: 'https://setosa.io/ev/markov-chains/' }]
    },

    // === SECURITY ===
    'aes': {
        id: 'aes',
        name: 'AES',
        fullName: 'Advanced Encryption Standard',
        category: 'Security',
        introduction: `AES is a symmetric block cipher for encrypting data in 128-bit blocks with 128/192/256-bit keys.`,
        mathematicalBackground: [
            { title: 'Block Size', content: 'Fixed 128-bit blocks.', formula: '128 bits = 16 bytes' }
        ],
        pseudocode: `AES_ENCRYPT(plaintext, key):\nstate = plaintext_to_state(plaintext)\nAddRoundKey(state, key[0])\nfor round in 1 to Nr-1:\n  SubBytes, ShiftRows, MixColumns, AddRoundKey\nSubBytes, ShiftRows, AddRoundKey\nreturn state_to_ciphertext(state)`,
        complexity: { time: 'O(1) per block', space: 'O(key_size)' },
        whenToUse: ['Data encryption', 'Secure communication', 'Disk encryption'],
        whenNotToUse: ['Key exchange (use asymmetric)'],
        tips: ['Use GCM mode for authenticated encryption'],
        exampleDataset: `plaintext + secret key`,
        exampleOutput: `{"ciphertext": "0x3ad77bb40d7a3660"}`,
        references: [{ title: 'AES', url: 'https://en.wikipedia.org/wiki/Advanced_Encryption_Standard' }]
    },

    'rsa': {
        id: 'rsa',
        name: 'RSA',
        fullName: 'RSA Cryptosystem',
        category: 'Security',
        introduction: `RSA is an asymmetric algorithm using public-private key pairs for encryption and digital signatures.`,
        mathematicalBackground: [
            { title: 'Key Generation', content: 'Based on factorization difficulty.', formula: 'n = p × q, e × d ≡ 1 (mod φ(n))' },
            { title: 'Encryption', content: 'Modular exponentiation.', formula: 'c = mᵉ mod n' }
        ],
        pseudocode: `RSA_KEYGEN(bits):\np, q = random_primes(bits/2)\nn = p * q\nφ = (p-1)(q-1)\ne = 65537\nd = mod_inverse(e, φ)\nreturn (n, e), (n, d)`,
        complexity: { time: 'O(k³) for k-bit keys', space: 'O(key_size)' },
        whenToUse: ['Key exchange', 'Digital signatures', 'TLS handshake'],
        whenNotToUse: ['Bulk data encryption (too slow)'],
        tips: ['Use 2048+ bit keys', 'Combine with AES for hybrid encryption'],
        exampleDataset: `message + public key`,
        exampleOutput: `{"ciphertext": "encrypted_value"}`,
        references: [{ title: 'RSA', url: 'https://en.wikipedia.org/wiki/RSA_(cryptosystem)' }]
    },

    'sha-256': {
        id: 'sha-256',
        name: 'SHA-256',
        fullName: 'Secure Hash Algorithm 256-bit',
        category: 'Security',
        introduction: `SHA-256 is a cryptographic hash producing a 256-bit digest for data integrity verification.`,
        mathematicalBackground: [
            { title: 'Output Size', content: 'Fixed 256-bit output.', formula: '256 bits = 64 hex chars' }
        ],
        pseudocode: `SHA256(message):\npreprocess (pad to 512-bit blocks)\ninitialize hash values (H0-H7)\nfor each block:\n  extend to 64 words\n  compression function\nreturn H0||H1||...||H7`,
        complexity: { time: 'O(n)', space: 'O(1)' },
        whenToUse: ['Data integrity', 'Digital signatures', 'Blockchain'],
        whenNotToUse: ['Password storage (use bcrypt)'],
        tips: ['One-way function', 'Add salt for password hashing'],
        exampleDataset: `"Hello World"`,
        exampleOutput: `{"hash": "a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e"}`,
        references: [{ title: 'SHA-256', url: 'https://en.wikipedia.org/wiki/SHA-2' }]
    },

    'hmac': {
        id: 'hmac',
        name: 'HMAC',
        fullName: 'Hash-based Message Authentication Code',
        category: 'Security',
        introduction: `HMAC combines a hash function with a secret key for message authentication.`,
        mathematicalBackground: [
            { title: 'HMAC Formula', content: 'Nested hashing with key padding.', formula: 'HMAC(K,m) = H((K⊕opad)||H((K⊕ipad)||m))' }
        ],
        pseudocode: `HMAC(key, message, hash):\nipad = 0x36 repeated\nopad = 0x5c repeated\nk_ipad = key XOR ipad\nk_opad = key XOR opad\nreturn hash(k_opad || hash(k_ipad || message))`,
        complexity: { time: 'O(n)', space: 'O(1)' },
        whenToUse: ['API authentication', 'JWT signing', 'Message verification'],
        whenNotToUse: ['Encryption needed'],
        tips: ['Use SHA-256 or stronger'],
        exampleDataset: `message + secret key`,
        exampleOutput: `{"hmac": "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"}`,
        references: [{ title: 'HMAC', url: 'https://en.wikipedia.org/wiki/HMAC' }]
    },

    'diffie-hellman': {
        id: 'diffie-hellman',
        name: 'Diffie-Hellman',
        fullName: 'Diffie-Hellman Key Exchange',
        category: 'Security',
        introduction: `Diffie-Hellman enables two parties to establish a shared secret over an insecure channel.`,
        mathematicalBackground: [
            { title: 'Key Exchange', content: 'Based on discrete logarithm problem.', formula: 's = gᵃᵇ mod p' }
        ],
        pseudocode: `DH_EXCHANGE(p, g):\na = random_private()\nA = g^a mod p  // public\n// send A, receive B\nshared_secret = B^a mod p`,
        complexity: { time: 'O(k³) for k-bit primes', space: 'O(k)' },
        whenToUse: ['Key exchange', 'TLS/SSL', 'VPN'],
        whenNotToUse: ['Without authentication (vulnerable to MITM)'],
        tips: ['Use 2048+ bit primes', 'ECDH is more efficient'],
        exampleDataset: `public parameters p, g`,
        exampleOutput: `{"sharedSecret": "same_value_for_both_parties"}`,
        references: [{ title: 'Diffie-Hellman', url: 'https://en.wikipedia.org/wiki/Diffie%E2%80%93Hellman_key_exchange' }]
    },

    // === RECOMMENDATION ===
    'collaborative-filtering': {
        id: 'collaborative-filtering',
        name: 'Collaborative Filtering',
        fullName: 'Collaborative Filtering Recommendation',
        category: 'Recommendation',
        introduction: `Recommends items based on preferences of similar users or similar items.`,
        mathematicalBackground: [
            { title: 'User Similarity', content: 'Cosine or Pearson similarity.', formula: 'sim(u,v) = cos(rᵤ, rᵥ)' },
            { title: 'Matrix Factorization', content: 'Decompose ratings matrix.', formula: 'R ≈ U × Vᵀ' }
        ],
        pseudocode: `RECOMMEND(user, ratings, k):\nsimilar_users = find_k_nearest(user, ratings, k)\ncandidates = union(items_rated_by(similar_users)) - items_rated_by(user)\nscores = weighted_average(ratings, similarity)\nreturn top_n(candidates, by=scores)`,
        complexity: { time: 'O(users × items)', space: 'O(users × items)' },
        whenToUse: ['Movie/music recommendations', 'E-commerce'],
        whenNotToUse: ['Cold start problem', 'Sparse data'],
        tips: ['Use matrix factorization for scale'],
        exampleDataset: `user-item ratings matrix`,
        exampleOutput: `{"recommendations": ["Movie A", "Movie B"]}`,
        references: [{ title: 'Collaborative Filtering', url: 'https://en.wikipedia.org/wiki/Collaborative_filtering' }]
    },

    'content-based-filtering': {
        id: 'content-based-filtering',
        name: 'Content-Based Filtering',
        fullName: 'Content-Based Recommendation',
        category: 'Recommendation',
        introduction: `Recommends items similar to those a user liked based on item features.`,
        mathematicalBackground: [
            { title: 'Similarity', content: 'Compare item feature vectors.', formula: 'score = cos(userProfile, itemFeatures)' }
        ],
        pseudocode: `RECOMMEND(user_profile, items):\nscores = []\nfor item in items:\n  score = cosine_similarity(user_profile, item.features)\n  scores.append((item, score))\nreturn sorted(scores, by=score, descending)`,
        complexity: { time: 'O(items × features)', space: 'O(features)' },
        whenToUse: ['News recommendations', 'New items'],
        whenNotToUse: ['Discovery of diverse content'],
        tips: ['Use TF-IDF for text features'],
        exampleDataset: `user profile + item features`,
        exampleOutput: `{"recommendations": ["Article A", "Article B"]}`,
        references: [{ title: 'Content-Based Filtering', url: 'https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering' }]
    },

    'lru-cache': {
        id: 'lru-cache',
        name: 'LRU Cache',
        fullName: 'Least Recently Used Cache',
        category: 'Recommendation',
        introduction: `LRU evicts least recently used items when cache is full. Uses hash map + doubly linked list for O(1) operations.`,
        mathematicalBackground: [
            { title: 'Operations', content: 'Constant time get and put.', formula: 'O(1) get/put' }
        ],
        pseudocode: `LRU_GET(key):\nif key in hashmap:\n  move to front of list\n  return value\nreturn -1\n\nLRU_PUT(key, value):\nif key in hashmap:\n  update value, move to front\nelse:\n  if at capacity: evict tail\n  insert at front`,
        complexity: { time: 'O(1)', space: 'O(capacity)' },
        whenToUse: ['Web caching', 'Database buffering', 'API caching'],
        whenNotToUse: ['Frequency matters more (use LFU)'],
        tips: ['Use OrderedDict in Python', 'Map + LinkedList in other languages'],
        exampleDataset: `cache capacity = 2, operations: put(1,1), put(2,2), get(1), put(3,3)`,
        exampleOutput: `{"get(1)": 1, "cache": {1: 1, 3: 3}}`,
        references: [{ title: 'LRU Cache', url: 'https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU' }]
    }
};
