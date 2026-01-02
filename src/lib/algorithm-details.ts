// Extended algorithm details for detail pages
export interface AlgorithmDetail {
    id: string;
    name: string;
    fullName: string;
    category: string;
    tagline: string;
    overview: string;
    howItWorks: {
        steps: string[];
        formula?: string;
    };
    prosAndCons: {
        pros: string[];
        cons: string[];
    };
    hyperparameters: {
        name: string;
        description: string;
        typical: string;
    }[];
    useCases: {
        title: string;
        description: string;
    }[];
    codeExample: string;
    relatedAlgorithms: string[];
}

export const algorithmDetails: Record<string, AlgorithmDetail> = {
    'k-means': {
        id: 'k-means',
        name: 'K-Means',
        fullName: 'K-Means Clustering',
        category: 'Clustering',
        tagline: 'Partitioning data into K distinct, non-overlapping clusters',
        overview: `K-Means is one of the simplest and most popular unsupervised machine learning algorithms. It aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centroid).`,
        howItWorks: {
            steps: [
                'Initialize K centroids randomly from the data points',
                'Assign each data point to the nearest centroid (using Euclidean distance)',
                'Recalculate centroids as the mean of all points assigned to that cluster',
                'Repeat steps 2-3 until centroids no longer change (convergence)'
            ],
            formula: 'J = Σᵢ₌₁ⁿ Σⱼ₌₁ᵏ wᵢⱼ ||xᵢ - μⱼ||²'
        },
        prosAndCons: {
            pros: [
                'Simple to understand and implement',
                'Scales well to large datasets',
                'Fast convergence in most cases',
                'Works well with spherical clusters'
            ],
            cons: [
                'Must specify K in advance',
                'Sensitive to initial centroid placement',
                'Struggles with non-spherical clusters',
                'Sensitive to outliers'
            ]
        },
        hyperparameters: [
            { name: 'K (n_clusters)', description: 'Number of clusters to form', typical: '2-10, use elbow method' },
            { name: 'max_iter', description: 'Maximum iterations for convergence', typical: '100-300' },
            { name: 'init', description: 'Initialization method', typical: 'k-means++ (recommended)' }
        ],
        useCases: [
            { title: 'Customer Segmentation', description: 'Group customers by purchasing behavior for targeted marketing' },
            { title: 'Image Compression', description: 'Reduce colors in an image by clustering similar pixels' },
            { title: 'Document Clustering', description: 'Organize documents into topic groups' }
        ],
        codeExample: `// Using K-Means with ALLgorithm
const result = await fetch('/api/run/kmeans', {
  method: 'POST',
  body: JSON.stringify({
    datasetId: 'customers',
    k: 3,
    maxIterations: 100,
    initialization: 'kmeans++'
  })
});`,
        relatedAlgorithms: ['dbscan', 'pca']
    },

    'naive-bayes': {
        id: 'naive-bayes',
        name: 'Naive Bayes',
        fullName: 'Gaussian Naive Bayes Classifier',
        category: 'Classification',
        tagline: 'Probabilistic classifier based on Bayes theorem',
        overview: `Naive Bayes is a probabilistic machine learning algorithm based on Bayes' Theorem. It's called "naive" because it assumes that features are independent of each other given the class label, which is rarely true in real-world data but works surprisingly well in practice.`,
        howItWorks: {
            steps: [
                'Calculate the prior probability of each class',
                'Calculate the likelihood of each feature given each class',
                'Apply Bayes theorem to compute posterior probability',
                'Assign the class with highest posterior probability'
            ],
            formula: 'P(C|X) = P(X|C) × P(C) / P(X)'
        },
        prosAndCons: {
            pros: [
                'Very fast training and prediction',
                'Works well with small datasets',
                'Handles high-dimensional data well',
                'Good baseline for text classification'
            ],
            cons: [
                'Assumes feature independence (rarely true)',
                'Struggles with features that have strong correlations',
                'Zero probability problem with unseen features',
                'Not ideal for regression tasks'
            ]
        },
        hyperparameters: [
            { name: 'var_smoothing', description: 'Portion of largest variance added for stability', typical: '1e-9' },
            { name: 'priors', description: 'Prior probabilities of classes', typical: 'None (calculated from data)' }
        ],
        useCases: [
            { title: 'Spam Detection', description: 'Classify emails as spam or not spam' },
            { title: 'Sentiment Analysis', description: 'Determine positive/negative sentiment in text' },
            { title: 'Medical Diagnosis', description: 'Predict disease presence based on symptoms' }
        ],
        codeExample: `// Using Naive Bayes with ALLgorithm
const result = await fetch('/api/run/naive-bayes', {
  method: 'POST',
  body: JSON.stringify({
    datasetId: 'titanic',
    trainRatio: 0.8
  })
});`,
        relatedAlgorithms: ['logistic-regression', 'svm']
    },

    'logistic-regression': {
        id: 'logistic-regression',
        name: 'Logistic Regression',
        fullName: 'Logistic Regression Classifier',
        category: 'Classification',
        tagline: 'Predicting probability of binary outcomes',
        overview: `Despite its name, Logistic Regression is a classification algorithm, not regression. It models the probability that an instance belongs to a particular class using the logistic (sigmoid) function, making it ideal for binary classification problems.`,
        howItWorks: {
            steps: [
                'Apply linear function: z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ',
                'Apply sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ)',
                'Output probability between 0 and 1',
                'Apply threshold (usually 0.5) to classify'
            ],
            formula: 'P(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))'
        },
        prosAndCons: {
            pros: [
                'Outputs probabilities, not just classes',
                'Highly interpretable coefficients',
                'Less prone to overfitting',
                'Works well with linearly separable data'
            ],
            cons: [
                'Assumes linear relationship',
                'Cannot handle complex non-linear relationships',
                'Sensitive to outliers',
                'Requires feature scaling for gradient descent'
            ]
        },
        hyperparameters: [
            { name: 'C', description: 'Inverse of regularization strength', typical: '1.0' },
            { name: 'max_iter', description: 'Maximum iterations for solver', typical: '100-1000' },
            { name: 'penalty', description: 'Regularization type (l1, l2)', typical: 'l2' }
        ],
        useCases: [
            { title: 'Credit Scoring', description: 'Predict loan default probability' },
            { title: 'Churn Prediction', description: 'Identify customers likely to leave' },
            { title: 'Click-Through Rate', description: 'Predict ad click probability' }
        ],
        codeExample: `// Logistic Regression example
// Coming soon to ALLgorithm!
// Will support binary classification
// with probability outputs`,
        relatedAlgorithms: ['naive-bayes', 'svm', 'linear-regression']
    },

    'linear-regression': {
        id: 'linear-regression',
        name: 'Linear Regression',
        fullName: 'Ordinary Least Squares Linear Regression',
        category: 'Regression',
        tagline: 'Finding the best-fit line through data points',
        overview: `Linear Regression is the most fundamental regression algorithm. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation. The goal is to minimize the sum of squared residuals between predicted and actual values.`,
        howItWorks: {
            steps: [
                'Assume a linear relationship: y = β₀ + β₁x₁ + ... + βₙxₙ + ε',
                'Calculate coefficients that minimize sum of squared errors',
                'Use Normal Equation or Gradient Descent to find optimal coefficients',
                'Make predictions using the fitted line/plane'
            ],
            formula: 'ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ'
        },
        prosAndCons: {
            pros: [
                'Simple and interpretable',
                'Fast to train',
                'No hyperparameters (vanilla version)',
                'Works well when relationship is actually linear'
            ],
            cons: [
                'Assumes linear relationship',
                'Sensitive to outliers',
                'Cannot capture complex patterns',
                'Assumes homoscedasticity'
            ]
        },
        hyperparameters: [
            { name: 'fit_intercept', description: 'Whether to calculate intercept', typical: 'True' },
            { name: 'normalize', description: 'Whether to normalize features', typical: 'False' }
        ],
        useCases: [
            { title: 'House Price Prediction', description: 'Predict prices based on features like size, rooms' },
            { title: 'Sales Forecasting', description: 'Predict future sales from historical trends' },
            { title: 'Risk Assessment', description: 'Estimate risk scores based on variables' }
        ],
        codeExample: `// Linear Regression example
// Coming soon to ALLgorithm!
// Will support simple and multiple
// linear regression with R² score`,
        relatedAlgorithms: ['logistic-regression', 'pca']
    },

    'pca': {
        id: 'pca',
        name: 'PCA',
        fullName: 'Principal Component Analysis',
        category: 'Dimensionality Reduction',
        tagline: 'Reducing dimensions while preserving variance',
        overview: `PCA is a statistical technique that transforms high-dimensional data into a lower-dimensional space. It finds the directions (principal components) that maximize variance in the data, allowing you to compress information while losing minimal information.`,
        howItWorks: {
            steps: [
                'Standardize the data (zero mean, unit variance)',
                'Compute the covariance matrix',
                'Calculate eigenvalues and eigenvectors',
                'Sort eigenvectors by eigenvalue (descending)',
                'Select top K eigenvectors as principal components',
                'Transform data into new coordinate system'
            ],
            formula: 'Z = X × W (where W contains eigenvectors)'
        },
        prosAndCons: {
            pros: [
                'Reduces dimensionality effectively',
                'Removes multicollinearity',
                'Speeds up other algorithms',
                'Helps visualize high-dimensional data'
            ],
            cons: [
                'Principal components are not interpretable',
                'Sensitive to feature scaling',
                'Assumes linear relationships',
                'Information loss if too few components'
            ]
        },
        hyperparameters: [
            { name: 'n_components', description: 'Number of components to keep', typical: '2-3 for visualization, 0.95 for variance' },
            { name: 'whiten', description: 'Whether to whiten components', typical: 'False' }
        ],
        useCases: [
            { title: 'Data Visualization', description: 'Plot high-dimensional data in 2D/3D' },
            { title: 'Noise Reduction', description: 'Remove noise by keeping major components' },
            { title: 'Feature Extraction', description: 'Create new features from existing ones' }
        ],
        codeExample: `// Using PCA with ALLgorithm
const result = await fetch('/api/run/pca', {
  method: 'POST',
  body: JSON.stringify({
    datasetId: 'iris',
    nComponents: 2
  })
});`,
        relatedAlgorithms: ['k-means', 'linear-regression']
    },

    'tf-idf-nb': {
        id: 'tf-idf-nb',
        name: 'TF-IDF + NB',
        fullName: 'TF-IDF with Naive Bayes',
        category: 'Text Mining',
        tagline: 'Text classification using word importance weights',
        overview: `This combination uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors, then applies Naive Bayes for classification. TF-IDF weighs words by how important they are to a document relative to the corpus.`,
        howItWorks: {
            steps: [
                'Tokenize documents into words',
                'Calculate Term Frequency (TF) for each word in each document',
                'Calculate Inverse Document Frequency (IDF) across corpus',
                'Multiply TF × IDF to get final weights',
                'Train Naive Bayes on TF-IDF vectors',
                'Classify new documents using trained model'
            ],
            formula: 'TF-IDF(t,d) = TF(t,d) × log(N / DF(t))'
        },
        prosAndCons: {
            pros: [
                'Handles text data naturally',
                'Penalizes common words automatically',
                'Fast and efficient',
                'Good baseline for text classification'
            ],
            cons: [
                'Ignores word order and context',
                'Vocabulary can become very large',
                'Cannot handle out-of-vocabulary words',
                'Does not capture semantic meaning'
            ]
        },
        hyperparameters: [
            { name: 'max_features', description: 'Maximum vocabulary size', typical: '5000-10000' },
            { name: 'ngram_range', description: 'Range of n-grams to use', typical: '(1, 2) for unigrams+bigrams' },
            { name: 'min_df', description: 'Minimum document frequency', typical: '2-5' }
        ],
        useCases: [
            { title: 'Document Classification', description: 'Categorize documents into topics' },
            { title: 'Sentiment Analysis', description: 'Classify reviews as positive/negative' },
            { title: 'News Categorization', description: 'Auto-tag news articles by category' }
        ],
        codeExample: `// TF-IDF + Naive Bayes example
// Coming soon to ALLgorithm!
// Will support text classification
// with customizable preprocessing`,
        relatedAlgorithms: ['naive-bayes', 'svm']
    },

    'dbscan': {
        id: 'dbscan',
        name: 'DBSCAN',
        fullName: 'Density-Based Spatial Clustering of Applications with Noise',
        category: 'Clustering',
        tagline: 'Finding clusters of arbitrary shape based on density',
        overview: `DBSCAN is a density-based clustering algorithm that groups together points that are closely packed, marking points in low-density regions as outliers. Unlike K-Means, it doesn't require specifying the number of clusters and can find arbitrarily shaped clusters.`,
        howItWorks: {
            steps: [
                'Pick an unvisited point',
                'Find all points within ε (epsilon) distance',
                'If ≥ MinPts neighbors, start a cluster',
                'Expand cluster by adding density-reachable points',
                'Mark low-density points as noise/outliers',
                'Repeat until all points are visited'
            ],
            formula: 'Core point: |Nε(p)| ≥ MinPts'
        },
        prosAndCons: {
            pros: [
                'No need to specify number of clusters',
                'Can find arbitrarily shaped clusters',
                'Robust to outliers (marks them as noise)',
                'Only two parameters to tune'
            ],
            cons: [
                'Sensitive to ε and MinPts parameters',
                'Struggles with varying density clusters',
                'Does not work well in high dimensions',
                'Not deterministic for border points'
            ]
        },
        hyperparameters: [
            { name: 'eps (ε)', description: 'Maximum distance between neighbors', typical: 'Use k-distance graph' },
            { name: 'min_samples', description: 'Minimum points to form dense region', typical: '5-10' }
        ],
        useCases: [
            { title: 'Anomaly Detection', description: 'Find outliers in transaction data' },
            { title: 'Geospatial Clustering', description: 'Identify hotspots on maps' },
            { title: 'Image Segmentation', description: 'Group pixels by color/position' }
        ],
        codeExample: `// DBSCAN example
// Coming soon to ALLgorithm!
// Will support anomaly detection
// with automatic noise labeling`,
        relatedAlgorithms: ['k-means', 'pca']
    },

    'svm': {
        id: 'svm',
        name: 'SVM',
        fullName: 'Support Vector Machine',
        category: 'Classification',
        tagline: 'Finding the optimal hyperplane to separate classes',
        overview: `SVM is a powerful supervised learning algorithm that finds the hyperplane which best separates different classes. It maximizes the margin between the closest points (support vectors) of different classes, making it robust and effective for high-dimensional spaces.`,
        howItWorks: {
            steps: [
                'Plot data points in n-dimensional space',
                'Find the hyperplane that separates classes',
                'Maximize margin between hyperplane and nearest points',
                'Points on the margin boundary are "support vectors"',
                'Use kernel trick for non-linear boundaries'
            ],
            formula: 'w·x + b = 0 (decision boundary)'
        },
        prosAndCons: {
            pros: [
                'Effective in high dimensions',
                'Memory efficient (uses only support vectors)',
                'Versatile with different kernels',
                'Good generalization with proper tuning'
            ],
            cons: [
                'Slow training on large datasets',
                'Sensitive to feature scaling',
                'Difficult to interpret',
                'Choosing the right kernel is tricky'
            ]
        },
        hyperparameters: [
            { name: 'C', description: 'Regularization parameter', typical: '1.0' },
            { name: 'kernel', description: 'Kernel type (linear, rbf, poly)', typical: 'rbf' },
            { name: 'gamma', description: 'Kernel coefficient', typical: 'scale or auto' }
        ],
        useCases: [
            { title: 'Image Classification', description: 'Classify images into categories' },
            { title: 'Handwriting Recognition', description: 'Recognize handwritten digits' },
            { title: 'Bioinformatics', description: 'Classify proteins or genes' }
        ],
        codeExample: `// SVM example
// Coming soon to ALLgorithm!
// Will support multiple kernels
// linear, RBF, polynomial`,
        relatedAlgorithms: ['naive-bayes', 'logistic-regression']
    },

    'decision-tree': {
        id: 'decision-tree',
        name: 'Decision Tree',
        fullName: 'Decision Tree Classifier',
        category: 'Classification',
        tagline: 'Making decisions through a tree of questions',
        overview: `Decision Tree is a supervised learning algorithm that creates a tree-like model of decisions. Each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label or decision.`,
        howItWorks: {
            steps: [
                'Start with the entire dataset at the root',
                'Select the best feature to split the data (using Gini or Entropy)',
                'Create branches for each possible value of the selected feature',
                'Recursively repeat for each branch until stopping criteria met',
                'Assign class labels to leaf nodes'
            ],
            formula: 'Gini = 1 - Σᵢ pᵢ² or Entropy = -Σᵢ pᵢ log₂(pᵢ)'
        },
        prosAndCons: {
            pros: [
                'Easy to understand and interpret',
                'Handles both numerical and categorical data',
                'Requires little data preprocessing',
                'Can capture non-linear relationships'
            ],
            cons: [
                'Prone to overfitting',
                'Sensitive to small changes in data',
                'Can create biased trees if classes are imbalanced',
                'May not generalize well'
            ]
        },
        hyperparameters: [
            { name: 'max_depth', description: 'Maximum depth of the tree', typical: '3-20' },
            { name: 'min_samples_split', description: 'Minimum samples to split a node', typical: '2-10' },
            { name: 'criterion', description: 'Split quality measure (gini/entropy)', typical: 'gini' }
        ],
        useCases: [
            { title: 'Medical Diagnosis', description: 'Diagnose diseases based on symptoms' },
            { title: 'Credit Approval', description: 'Decide loan approval based on applicant features' },
            { title: 'Customer Churn', description: 'Predict which customers will leave' }
        ],
        codeExample: `// Decision Tree example
// Coming soon to ALLgorithm!
// Will support classification
// with interpretable rule extraction`,
        relatedAlgorithms: ['random-forest', 'naive-bayes']
    },

    'random-forest': {
        id: 'random-forest',
        name: 'Random Forest',
        fullName: 'Random Forest Ensemble Classifier',
        category: 'Classification',
        tagline: 'Ensemble of decision trees for robust predictions',
        overview: `Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of individual trees. It uses bagging and random feature selection to reduce overfitting.`,
        howItWorks: {
            steps: [
                'Create multiple bootstrap samples from the dataset',
                'Build a decision tree for each sample',
                'At each split, consider only a random subset of features',
                'Aggregate predictions from all trees',
                'Output the majority vote (classification) or average (regression)'
            ],
            formula: 'Prediction = mode(Tree₁, Tree₂, ..., Treeₙ)'
        },
        prosAndCons: {
            pros: [
                'Reduces overfitting compared to single tree',
                'Handles missing values well',
                'Provides feature importance rankings',
                'Works well with large datasets'
            ],
            cons: [
                'Less interpretable than single tree',
                'Can be slow for real-time predictions',
                'Requires more memory',
                'May overfit on noisy datasets'
            ]
        },
        hyperparameters: [
            { name: 'n_estimators', description: 'Number of trees in the forest', typical: '100-500' },
            { name: 'max_features', description: 'Features to consider at each split', typical: 'sqrt(n_features)' },
            { name: 'max_depth', description: 'Maximum depth of each tree', typical: 'None (unlimited)' }
        ],
        useCases: [
            { title: 'Fraud Detection', description: 'Identify fraudulent transactions' },
            { title: 'Stock Prediction', description: 'Predict stock price movements' },
            { title: 'Disease Prediction', description: 'Predict disease risk from patient data' }
        ],
        codeExample: `// Random Forest example
// Coming soon to ALLgorithm!
// Will support ensemble classification
// with feature importance analysis`,
        relatedAlgorithms: ['decision-tree', 'svm']
    },

    'knn': {
        id: 'knn',
        name: 'KNN',
        fullName: 'K-Nearest Neighbors',
        category: 'Classification',
        tagline: 'Classify based on the neighbors you keep',
        overview: `K-Nearest Neighbors is a simple, instance-based learning algorithm that classifies a data point based on how its neighbors are classified. It stores all available cases and classifies new cases based on a similarity measure (distance function).`,
        howItWorks: {
            steps: [
                'Store all training examples',
                'When a new point arrives, calculate distance to all training points',
                'Select the K nearest neighbors',
                'For classification: assign majority class of neighbors',
                'For regression: assign average value of neighbors'
            ],
            formula: 'd(x,y) = √(Σᵢ(xᵢ - yᵢ)²) (Euclidean distance)'
        },
        prosAndCons: {
            pros: [
                'Simple to understand and implement',
                'No training phase (lazy learning)',
                'Naturally handles multi-class cases',
                'Can capture complex decision boundaries'
            ],
            cons: [
                'Slow prediction for large datasets',
                'Sensitive to irrelevant features',
                'Requires feature scaling',
                'Memory intensive (stores all data)'
            ]
        },
        hyperparameters: [
            { name: 'k', description: 'Number of neighbors to consider', typical: '3-10, odd number' },
            { name: 'weights', description: 'Weight function (uniform/distance)', typical: 'uniform' },
            { name: 'metric', description: 'Distance metric', typical: 'euclidean' }
        ],
        useCases: [
            { title: 'Recommendation Systems', description: 'Recommend items based on similar users' },
            { title: 'Image Recognition', description: 'Classify images by comparing to known examples' },
            { title: 'Anomaly Detection', description: 'Detect outliers far from neighbors' }
        ],
        codeExample: `// KNN example
// Coming soon to ALLgorithm!
// Will support classification
// with customizable distance metrics`,
        relatedAlgorithms: ['naive-bayes', 'svm']
    },

    'hierarchical-clustering': {
        id: 'hierarchical-clustering',
        name: 'Hierarchical Clustering',
        fullName: 'Agglomerative Hierarchical Clustering',
        category: 'Clustering',
        tagline: 'Building a tree of clusters from bottom up',
        overview: `Hierarchical Clustering creates a tree of clusters called a dendrogram. In agglomerative (bottom-up) approach, each data point starts as its own cluster, and pairs of clusters are merged as one moves up the hierarchy based on their similarity.`,
        howItWorks: {
            steps: [
                'Start with each point as its own cluster',
                'Calculate distance between all pairs of clusters',
                'Merge the two closest clusters',
                'Update distance matrix',
                'Repeat until only one cluster remains',
                'Cut the dendrogram at desired level to get clusters'
            ],
            formula: 'd(A,B) = linkage(d(a,b)) for a∈A, b∈B'
        },
        prosAndCons: {
            pros: [
                'No need to specify number of clusters beforehand',
                'Produces a dendrogram for visualization',
                'Can capture hierarchical relationships',
                'Works with any distance metric'
            ],
            cons: [
                'Computationally expensive O(n³)',
                'Sensitive to noise and outliers',
                'Cannot undo a merge once made',
                'Difficult to apply to large datasets'
            ]
        },
        hyperparameters: [
            { name: 'linkage', description: 'How to measure cluster distance', typical: 'ward, complete, average' },
            { name: 'distance_threshold', description: 'Threshold for cutting dendrogram', typical: 'None' },
            { name: 'n_clusters', description: 'Number of clusters to find', typical: '2-10' }
        ],
        useCases: [
            { title: 'Taxonomy Creation', description: 'Build hierarchical categories of items' },
            { title: 'Gene Analysis', description: 'Group genes with similar expression patterns' },
            { title: 'Document Organization', description: 'Organize documents in hierarchical folders' }
        ],
        codeExample: `// Hierarchical Clustering example
// Coming soon to ALLgorithm!
// Will support dendrogram visualization
// with multiple linkage methods`,
        relatedAlgorithms: ['k-means', 'dbscan']
    },

    'quick-sort': {
        id: 'quick-sort',
        name: 'Quick Sort',
        fullName: 'Quick Sort Algorithm',
        category: 'Sorting',
        tagline: 'Divide and conquer with pivot partitioning',
        overview: `Quick Sort is one of the most efficient sorting algorithms. It works by selecting a 'pivot' element and partitioning the array around the pivot, putting all smaller elements before it and larger elements after it. This process is then applied recursively to the sub-arrays.`,
        howItWorks: {
            steps: [
                'Choose a pivot element from the array',
                'Partition: reorder so elements < pivot come before, > pivot come after',
                'Recursively apply to the sub-array before the pivot',
                'Recursively apply to the sub-array after the pivot',
                'Base case: arrays of size 0 or 1 are sorted'
            ],
            formula: 'T(n) = T(k) + T(n-k-1) + O(n)'
        },
        prosAndCons: {
            pros: [
                'Very fast average case O(n log n)',
                'In-place sorting (low memory)',
                'Cache-efficient',
                'Widely used in practice'
            ],
            cons: [
                'Worst case O(n²) with bad pivot choice',
                'Not stable (equal elements may reorder)',
                'Recursive (stack overhead)',
                'Performance depends on pivot selection'
            ]
        },
        hyperparameters: [
            { name: 'pivot_strategy', description: 'How to choose pivot', typical: 'median-of-three' },
            { name: 'threshold', description: 'Switch to insertion sort for small arrays', typical: '10-20' }
        ],
        useCases: [
            { title: 'General Sorting', description: 'Default sorting algorithm in many libraries' },
            { title: 'Database Operations', description: 'Sort query results efficiently' },
            { title: 'File Systems', description: 'Sort directory listings' }
        ],
        codeExample: `// Quick Sort visualization
// See the algorithm in action!
function quickSort(arr, low, high) {
  if (low < high) {
    let pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}`,
        relatedAlgorithms: ['merge-sort', 'bubble-sort']
    },

    'merge-sort': {
        id: 'merge-sort',
        name: 'Merge Sort',
        fullName: 'Merge Sort Algorithm',
        category: 'Sorting',
        tagline: 'Divide, sort, and merge back together',
        overview: `Merge Sort is a divide-and-conquer algorithm that divides the input array into two halves, recursively sorts them, and then merges the two sorted halves. It guarantees O(n log n) time complexity in all cases.`,
        howItWorks: {
            steps: [
                'Divide the array into two halves',
                'Recursively sort the left half',
                'Recursively sort the right half',
                'Merge the two sorted halves into one sorted array',
                'Base case: single element is already sorted'
            ],
            formula: 'T(n) = 2T(n/2) + O(n) = O(n log n)'
        },
        prosAndCons: {
            pros: [
                'Guaranteed O(n log n) in all cases',
                'Stable sort (preserves order of equal elements)',
                'Good for linked lists',
                'Parallelizable'
            ],
            cons: [
                'Requires O(n) extra space',
                'Slower than Quick Sort in practice for arrays',
                'Not in-place',
                'Overkill for small arrays'
            ]
        },
        hyperparameters: [
            { name: 'threshold', description: 'Switch to insertion sort for small arrays', typical: '10-20' }
        ],
        useCases: [
            { title: 'External Sorting', description: 'Sort data too large to fit in memory' },
            { title: 'Linked Lists', description: 'Efficient sorting of linked structures' },
            { title: 'Parallel Processing', description: 'Distribute sorting across processors' }
        ],
        codeExample: `// Merge Sort visualization
function mergeSort(arr) {
  if (arr.length <= 1) return arr;
  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  return merge(left, right);
}`,
        relatedAlgorithms: ['quick-sort', 'bubble-sort']
    },

    'bubble-sort': {
        id: 'bubble-sort',
        name: 'Bubble Sort',
        fullName: 'Bubble Sort Algorithm',
        category: 'Sorting',
        tagline: 'Simple swapping until sorted',
        overview: `Bubble Sort is the simplest sorting algorithm that works by repeatedly stepping through the list, comparing adjacent elements and swapping them if they are in the wrong order. The pass through the list is repeated until no swaps are needed.`,
        howItWorks: {
            steps: [
                'Start from the first element',
                'Compare current element with the next element',
                'If current > next, swap them',
                'Move to the next pair and repeat',
                'After each pass, the largest unsorted element bubbles to its correct position',
                'Repeat until no swaps occur in a pass'
            ],
            formula: 'T(n) = O(n²) comparisons and swaps'
        },
        prosAndCons: {
            pros: [
                'Very simple to understand and implement',
                'Stable sort',
                'In-place (no extra memory)',
                'Can detect already sorted array in O(n)'
            ],
            cons: [
                'Very slow O(n²)',
                'Not practical for large datasets',
                'More swaps than other algorithms',
                'Rarely used in production'
            ]
        },
        hyperparameters: [],
        useCases: [
            { title: 'Educational Purposes', description: 'Teaching sorting algorithm concepts' },
            { title: 'Small Datasets', description: 'When simplicity matters more than speed' },
            { title: 'Nearly Sorted Data', description: 'Efficient when data is almost sorted' }
        ],
        codeExample: `// Bubble Sort visualization
function bubbleSort(arr) {
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr.length - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
      }
    }
  }
  return arr;
}`,
        relatedAlgorithms: ['quick-sort', 'merge-sort']
    },

    'binary-search': {
        id: 'binary-search',
        name: 'Binary Search',
        fullName: 'Binary Search Algorithm',
        category: 'Searching',
        tagline: 'Finding elements by halving the search space',
        overview: `Binary Search is an efficient algorithm for finding an item in a sorted list. It works by repeatedly dividing the search interval in half. If the target value is less than the middle element, search the left half; otherwise, search the right half.`,
        howItWorks: {
            steps: [
                'Start with the entire sorted array',
                'Find the middle element',
                'If middle equals target, return index',
                'If target < middle, search left half',
                'If target > middle, search right half',
                'Repeat until found or search space is empty'
            ],
            formula: 'T(n) = O(log n)'
        },
        prosAndCons: {
            pros: [
                'Very fast O(log n) time complexity',
                'Simple to implement',
                'Space efficient O(1) for iterative',
                'Works great for large sorted datasets'
            ],
            cons: [
                'Requires sorted data',
                'Not efficient for small datasets',
                'Random access needed (not good for linked lists)',
                'Insertion/deletion may require re-sorting'
            ]
        },
        hyperparameters: [],
        useCases: [
            { title: 'Dictionary Lookup', description: 'Find words in a sorted dictionary' },
            { title: 'Database Indexing', description: 'Search in B-tree indexes' },
            { title: 'Version Control', description: 'Git bisect to find bugs' }
        ],
        codeExample: `// Binary Search visualization
function binarySearch(arr, target) {
  let left = 0, right = arr.length - 1;
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
}`,
        relatedAlgorithms: ['quick-sort', 'merge-sort']
    },

    'dijkstra': {
        id: 'dijkstra',
        name: 'Dijkstra',
        fullName: "Dijkstra's Shortest Path Algorithm",
        category: 'Graph',
        tagline: 'Finding the shortest path in weighted graphs',
        overview: `Dijkstra's algorithm finds the shortest paths from a source vertex to all other vertices in a weighted graph with non-negative edge weights. It uses a greedy approach, always expanding the vertex with the smallest known distance.`,
        howItWorks: {
            steps: [
                'Initialize distances: source = 0, all others = ∞',
                'Add source to priority queue',
                'While queue not empty: extract minimum distance vertex',
                'For each neighbor: if new path shorter, update distance',
                'Mark extracted vertex as visited',
                'Repeat until all vertices processed'
            ],
            formula: 'd[v] = min(d[v], d[u] + weight(u,v))'
        },
        prosAndCons: {
            pros: [
                'Finds optimal shortest path',
                'Works for directed and undirected graphs',
                'Efficient with priority queue O((V+E)log V)',
                'Widely applicable (maps, networks)'
            ],
            cons: [
                'Does not work with negative weights',
                'Processes all vertices (not goal-directed)',
                'Memory intensive for dense graphs',
                'May be slow for very large graphs'
            ]
        },
        hyperparameters: [
            { name: 'source', description: 'Starting vertex', typical: 'User specified' },
            { name: 'priority_queue', description: 'Data structure for efficiency', typical: 'Binary heap or Fibonacci heap' }
        ],
        useCases: [
            { title: 'GPS Navigation', description: 'Find shortest route between locations' },
            { title: 'Network Routing', description: 'Route packets through networks' },
            { title: 'Social Networks', description: 'Find degrees of separation' }
        ],
        codeExample: `// Dijkstra visualization
// Find shortest paths from source
// Uses priority queue for efficiency
// See the path unfold step by step!`,
        relatedAlgorithms: ['bfs-dfs']
    },

    'bfs-dfs': {
        id: 'bfs-dfs',
        name: 'BFS/DFS',
        fullName: 'Breadth-First Search / Depth-First Search',
        category: 'Graph',
        tagline: 'Exploring graphs systematically',
        overview: `BFS and DFS are fundamental graph traversal algorithms. BFS explores level by level using a queue, finding shortest paths in unweighted graphs. DFS explores as deep as possible using a stack (or recursion), useful for path finding and cycle detection.`,
        howItWorks: {
            steps: [
                'BFS: Start at source, add to queue',
                'BFS: Dequeue vertex, visit all unvisited neighbors, enqueue them',
                'BFS: Repeat until queue empty (level by level)',
                'DFS: Start at source, push to stack',
                'DFS: Pop vertex, visit it, push unvisited neighbors',
                'DFS: Repeat until stack empty (depth first)'
            ],
            formula: 'Time: O(V + E), Space: O(V)'
        },
        prosAndCons: {
            pros: [
                'BFS: Finds shortest path in unweighted graphs',
                'BFS: Good for finding nearby nodes first',
                'DFS: Uses less memory for deep graphs',
                'DFS: Good for path existence, cycle detection'
            ],
            cons: [
                'BFS: High memory for wide graphs',
                'BFS: Not suitable for weighted shortest path',
                'DFS: May not find shortest path',
                'DFS: Can get stuck in infinite paths without marking'
            ]
        },
        hyperparameters: [
            { name: 'start', description: 'Starting vertex for traversal', typical: 'User specified' },
            { name: 'goal', description: 'Target vertex (optional)', typical: 'None for full traversal' }
        ],
        useCases: [
            { title: 'Web Crawling', description: 'Explore web pages systematically' },
            { title: 'Maze Solving', description: 'Find path through a maze' },
            { title: 'Social Networks', description: 'Find friends of friends (BFS) or check connectivity' }
        ],
        codeExample: `// BFS visualization
function bfs(graph, start) {
  const queue = [start];
  const visited = new Set([start]);
  while (queue.length > 0) {
    const node = queue.shift();
    for (const neighbor of graph[node]) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
}`,
        relatedAlgorithms: ['dijkstra']
    },

    // === NEW SORTING ALGORITHMS ===
    'heap-sort': {
        id: 'heap-sort',
        name: 'Heap Sort',
        fullName: 'Heap Sort Algorithm',
        category: 'Sorting',
        tagline: 'Sorting with a binary heap data structure',
        overview: `Heap Sort uses a binary heap to sort elements. It first builds a max-heap from the data, then repeatedly extracts the maximum element and places it at the end of the sorted portion.`,
        howItWorks: {
            steps: [
                'Build a max-heap from the input array',
                'Swap the root (maximum) with the last element',
                'Reduce heap size by 1 and heapify the root',
                'Repeat until heap size is 1'
            ],
            formula: 'T(n) = O(n log n) in all cases'
        },
        prosAndCons: {
            pros: ['Guaranteed O(n log n)', 'In-place sorting', 'No worst case degradation', 'Good for priority queues'],
            cons: ['Not stable', 'Poor cache performance', 'Slower than Quick Sort in practice', 'More swaps than Merge Sort']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Priority Queue', description: 'Implement priority-based scheduling' },
            { title: 'External Sorting', description: 'Sort data larger than memory' },
            { title: 'Order Statistics', description: 'Find kth largest element' }
        ],
        codeExample: `function heapSort(arr) {\n  // Build max heap\n  for (let i = Math.floor(arr.length/2) - 1; i >= 0; i--)\n    heapify(arr, arr.length, i);\n  // Extract elements\n  for (let i = arr.length-1; i > 0; i--) {\n    [arr[0], arr[i]] = [arr[i], arr[0]];\n    heapify(arr, i, 0);\n  }\n}`,
        relatedAlgorithms: ['quick-sort', 'merge-sort']
    },

    'insertion-sort': {
        id: 'insertion-sort',
        name: 'Insertion Sort',
        fullName: 'Insertion Sort Algorithm',
        category: 'Sorting',
        tagline: 'Building sorted array one element at a time',
        overview: `Insertion Sort builds the final sorted array one item at a time. It takes each element and inserts it into its correct position among the already-sorted elements.`,
        howItWorks: {
            steps: [
                'Start from the second element',
                'Compare with elements before it',
                'Shift larger elements one position right',
                'Insert current element in correct position',
                'Repeat for all remaining elements'
            ],
            formula: 'T(n) = O(n²) worst/average, O(n) best'
        },
        prosAndCons: {
            pros: ['Simple implementation', 'Efficient for small data', 'Adaptive (fast for nearly sorted)', 'Stable sort', 'In-place'],
            cons: ['O(n²) time complexity', 'Slow for large datasets', 'Many shifts required']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Small Arrays', description: 'Efficient for n < 50' },
            { title: 'Nearly Sorted Data', description: 'Very fast when few elements out of place' },
            { title: 'Online Sorting', description: 'Sort data as it arrives' }
        ],
        codeExample: `function insertionSort(arr) {\n  for (let i = 1; i < arr.length; i++) {\n    let key = arr[i], j = i - 1;\n    while (j >= 0 && arr[j] > key) {\n      arr[j + 1] = arr[j];\n      j--;\n    }\n    arr[j + 1] = key;\n  }\n}`,
        relatedAlgorithms: ['bubble-sort', 'quick-sort']
    },

    // === NEW SEARCHING ALGORITHMS ===
    'linear-search': {
        id: 'linear-search',
        name: 'Linear Search',
        fullName: 'Linear Search Algorithm',
        category: 'Searching',
        tagline: 'Simple sequential search through data',
        overview: `Linear Search sequentially checks each element until a match is found or the end is reached. It's the simplest search algorithm and works on unsorted data.`,
        howItWorks: {
            steps: [
                'Start from the first element',
                'Compare current element with target',
                'If match found, return index',
                'Move to next element',
                'If end reached, return not found'
            ],
            formula: 'T(n) = O(n)'
        },
        prosAndCons: {
            pros: ['Works on unsorted data', 'Simple implementation', 'No preprocessing needed', 'Works on any data structure'],
            cons: ['O(n) time complexity', 'Inefficient for large datasets', 'No early termination guarantee']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Small Lists', description: 'When data is small, overhead of sorting not worth it' },
            { title: 'Unsorted Data', description: 'When sorting is not feasible' },
            { title: 'Single Search', description: 'One-time search where sorting cost > search cost' }
        ],
        codeExample: `function linearSearch(arr, target) {\n  for (let i = 0; i < arr.length; i++) {\n    if (arr[i] === target) return i;\n  }\n  return -1;\n}`,
        relatedAlgorithms: ['binary-search']
    },

    // === NEW GRAPH ALGORITHMS ===
    'a-star': {
        id: 'a-star',
        name: 'A*',
        fullName: 'A* Search Algorithm',
        category: 'Graph',
        tagline: 'Best-first search with heuristics',
        overview: `A* is a graph search algorithm that finds the shortest path using a heuristic to guide its search. It combines the actual cost from start with an estimated cost to goal.`,
        howItWorks: {
            steps: [
                'f(n) = g(n) + h(n) where g is actual cost, h is heuristic',
                'Maintain open set (frontier) and closed set (visited)',
                'Always expand node with lowest f(n)',
                'Update neighbors if better path found',
                'Stop when goal is reached or open set is empty'
            ],
            formula: 'f(n) = g(n) + h(n)'
        },
        prosAndCons: {
            pros: ['Optimal if heuristic is admissible', 'Faster than Dijkstra with good heuristic', 'Widely used in games', 'Flexible heuristic choice'],
            cons: ['Requires good heuristic', 'Memory intensive', 'Heuristic design is crucial']
        },
        hyperparameters: [
            { name: 'heuristic', description: 'Estimated cost to goal', typical: 'Euclidean or Manhattan distance' }
        ],
        useCases: [
            { title: 'Game Pathfinding', description: 'Character movement in games' },
            { title: 'Robot Navigation', description: 'Autonomous vehicle routing' },
            { title: 'Puzzle Solving', description: '8-puzzle, Rubik\'s cube' }
        ],
        codeExample: `// A* finds path using f(n) = g(n) + h(n)\n// g(n) = actual cost from start\n// h(n) = heuristic estimate to goal\n// Expands nodes with lowest f(n) first`,
        relatedAlgorithms: ['dijkstra', 'bfs-dfs']
    },

    'bellman-ford': {
        id: 'bellman-ford',
        name: 'Bellman-Ford',
        fullName: 'Bellman-Ford Algorithm',
        category: 'Graph',
        tagline: 'Shortest paths with negative weights',
        overview: `Bellman-Ford computes shortest paths from a source vertex, handling graphs with negative edge weights. It can also detect negative cycles.`,
        howItWorks: {
            steps: [
                'Initialize distances: source = 0, others = ∞',
                'Relax all edges V-1 times',
                'For each edge (u,v): if d[u] + w < d[v], update d[v]',
                'Check for negative cycles with one more iteration',
                'If any distance updates, negative cycle exists'
            ],
            formula: 'T(n) = O(V × E)'
        },
        prosAndCons: {
            pros: ['Handles negative weights', 'Detects negative cycles', 'Simple implementation', 'Works on directed graphs'],
            cons: ['Slower than Dijkstra O(VE)', 'Not suitable for large graphs', 'Cannot handle negative cycles for paths']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Currency Arbitrage', description: 'Detect profitable exchange cycles' },
            { title: 'Network Routing', description: 'Distance vector protocols' },
            { title: 'Negative Weights', description: 'When Dijkstra cannot be used' }
        ],
        codeExample: `// Bellman-Ford: relax all edges V-1 times\nfor (let i = 0; i < V-1; i++) {\n  for (const [u, v, w] of edges) {\n    if (dist[u] + w < dist[v])\n      dist[v] = dist[u] + w;\n  }\n}`,
        relatedAlgorithms: ['dijkstra', 'floyd-warshall']
    },

    'floyd-warshall': {
        id: 'floyd-warshall',
        name: 'Floyd-Warshall',
        fullName: 'Floyd-Warshall Algorithm',
        category: 'Graph',
        tagline: 'All-pairs shortest paths',
        overview: `Floyd-Warshall finds shortest paths between all pairs of vertices. It uses dynamic programming to consider all possible intermediate vertices.`,
        howItWorks: {
            steps: [
                'Initialize distance matrix from adjacency matrix',
                'For each intermediate vertex k',
                'For each pair (i, j): check if path through k is shorter',
                'Update: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])',
                'Final matrix contains all-pairs shortest distances'
            ],
            formula: 'T(n) = O(V³)'
        },
        prosAndCons: {
            pros: ['Finds all-pairs shortest paths', 'Handles negative edges', 'Simple implementation', 'Works on dense graphs'],
            cons: ['O(V³) time complexity', 'O(V²) space', 'Slow for sparse graphs', 'Cannot handle negative cycles']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Network Analysis', description: 'Find distances between all nodes' },
            { title: 'Transitive Closure', description: 'Check reachability between all pairs' },
            { title: 'Dense Graphs', description: 'When E ≈ V²' }
        ],
        codeExample: `// Floyd-Warshall: consider all intermediates\nfor (let k = 0; k < V; k++)\n  for (let i = 0; i < V; i++)\n    for (let j = 0; j < V; j++)\n      dist[i][j] = Math.min(dist[i][j],\n                            dist[i][k] + dist[k][j]);`,
        relatedAlgorithms: ['dijkstra', 'bellman-ford']
    },

    'pagerank': {
        id: 'pagerank',
        name: 'PageRank',
        fullName: 'PageRank Algorithm',
        category: 'Graph',
        tagline: 'Ranking nodes by link structure',
        overview: `PageRank, developed by Google founders, ranks nodes based on the structure of incoming links. A page is important if it's linked by other important pages.`,
        howItWorks: {
            steps: [
                'Initialize all pages with equal rank (1/N)',
                'For each iteration, distribute rank through outgoing links',
                'Apply damping factor (typically 0.85)',
                'PR(A) = (1-d)/N + d × Σ(PR(Ti)/C(Ti))',
                'Iterate until convergence'
            ],
            formula: 'PR(A) = (1-d)/N + d × Σ PR(Ti)/C(Ti)'
        },
        prosAndCons: {
            pros: ['Measures node importance', 'Scalable to large graphs', 'Intuitive interpretation', 'Foundation of web search'],
            cons: ['Vulnerable to manipulation', 'Slow convergence on large graphs', 'Requires iterative computation']
        },
        hyperparameters: [
            { name: 'damping_factor', description: 'Probability of following links', typical: '0.85' },
            { name: 'iterations', description: 'Number of iterations', typical: '100 or until convergence' }
        ],
        useCases: [
            { title: 'Web Search', description: 'Rank web pages by importance' },
            { title: 'Social Networks', description: 'Identify influential users' },
            { title: 'Citation Networks', description: 'Rank academic papers' }
        ],
        codeExample: `// PageRank iteration\nfor (let iter = 0; iter < maxIter; iter++) {\n  newRank = (1 - d) / N;\n  for (const [from, to] of links) {\n    newRank[to] += d * rank[from] / outDegree[from];\n  }\n  rank = newRank;\n}`,
        relatedAlgorithms: ['bfs-dfs']
    },

    // === DEEP LEARNING ===
    'neural-network': {
        id: 'neural-network',
        name: 'Neural Network',
        fullName: 'Artificial Neural Network',
        category: 'Deep Learning',
        tagline: 'Learning through interconnected neurons',
        overview: `Artificial Neural Networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that learn to recognize patterns through training.`,
        howItWorks: {
            steps: [
                'Input layer receives features',
                'Forward pass: compute weighted sums + activations',
                'Hidden layers transform representations',
                'Output layer produces predictions',
                'Backpropagation updates weights to minimize loss'
            ],
            formula: 'output = activation(Σ wᵢxᵢ + b)'
        },
        prosAndCons: {
            pros: ['Universal function approximator', 'Learns complex patterns', 'Highly flexible architecture', 'State-of-the-art in many domains'],
            cons: ['Requires large data', 'Computationally expensive', 'Black box (less interpretable)', 'Many hyperparameters to tune']
        },
        hyperparameters: [
            { name: 'layers', description: 'Number and size of hidden layers', typical: '2-5 layers' },
            { name: 'learning_rate', description: 'Step size for gradient descent', typical: '0.001' },
            { name: 'activation', description: 'Activation function', typical: 'ReLU, sigmoid, tanh' }
        ],
        useCases: [
            { title: 'Image Recognition', description: 'Classify images into categories' },
            { title: 'Speech Recognition', description: 'Convert speech to text' },
            { title: 'Prediction', description: 'Forecast time series, prices' }
        ],
        codeExample: `// Simple neural network forward pass\nfunction forward(x, weights) {\n  let h = relu(matmul(x, weights[0]));\n  for (let i = 1; i < weights.length-1; i++)\n    h = relu(matmul(h, weights[i]));\n  return softmax(matmul(h, weights.last));\n}`,
        relatedAlgorithms: ['cnn', 'rnn-lstm']
    },

    'cnn': {
        id: 'cnn',
        name: 'CNN',
        fullName: 'Convolutional Neural Network',
        category: 'Deep Learning',
        tagline: 'Neural networks for image processing',
        overview: `CNNs are specialized neural networks designed for processing grid-like data such as images. They use convolutional layers to automatically learn spatial hierarchies of features.`,
        howItWorks: {
            steps: [
                'Convolutional layers apply filters to detect features',
                'Pooling layers reduce spatial dimensions',
                'Multiple conv-pool blocks extract hierarchical features',
                'Flatten and connect to fully connected layers',
                'Output classification or detection results'
            ],
            formula: 'Conv: (f * g)(t) = Σ f(τ)g(t-τ)'
        },
        prosAndCons: {
            pros: ['Automatic feature extraction', 'Translation invariant', 'Parameter sharing reduces complexity', 'State-of-the-art for images'],
            cons: ['Requires GPU for training', 'Large memory footprint', 'Many hyperparameters', 'Needs lots of training data']
        },
        hyperparameters: [
            { name: 'filters', description: 'Number of convolutional filters', typical: '32, 64, 128' },
            { name: 'kernel_size', description: 'Size of convolutional kernel', typical: '3x3 or 5x5' },
            { name: 'pooling', description: 'Pooling type and size', typical: 'Max pooling 2x2' }
        ],
        useCases: [
            { title: 'Image Classification', description: 'Identify objects in images' },
            { title: 'Object Detection', description: 'Locate objects with bounding boxes' },
            { title: 'Face Recognition', description: 'Identify and verify faces' }
        ],
        codeExample: `// CNN architecture concept\nmodel = Sequential([\n  Conv2D(32, 3, activation='relu'),\n  MaxPooling2D(2),\n  Conv2D(64, 3, activation='relu'),\n  MaxPooling2D(2),\n  Flatten(),\n  Dense(10, activation='softmax')\n])`,
        relatedAlgorithms: ['neural-network', 'transformer']
    },

    'rnn-lstm': {
        id: 'rnn-lstm',
        name: 'RNN/LSTM',
        fullName: 'Recurrent Neural Network / Long Short-Term Memory',
        category: 'Deep Learning',
        tagline: 'Neural networks with memory for sequences',
        overview: `RNNs process sequential data by maintaining hidden state across time steps. LSTMs add memory cells and gates to handle long-term dependencies and avoid vanishing gradients.`,
        howItWorks: {
            steps: [
                'Process input sequence one element at a time',
                'Hidden state carries information across steps',
                'LSTM uses forget, input, output gates',
                'Cell state allows long-term memory',
                'Output can be sequence or final prediction'
            ],
            formula: 'hₜ = f(Wₓxₜ + Wₕhₜ₋₁ + b)'
        },
        prosAndCons: {
            pros: ['Handles variable-length sequences', 'Captures temporal dependencies', 'LSTM solves vanishing gradient', 'Good for time series'],
            cons: ['Slow training (sequential)', 'Difficult to parallelize', 'Can still struggle with very long sequences', 'Complex architecture']
        },
        hyperparameters: [
            { name: 'hidden_size', description: 'Size of hidden state', typical: '128, 256, 512' },
            { name: 'num_layers', description: 'Number of stacked layers', typical: '1-3' },
            { name: 'dropout', description: 'Dropout rate', typical: '0.2-0.5' }
        ],
        useCases: [
            { title: 'Language Modeling', description: 'Predict next word in sequence' },
            { title: 'Time Series', description: 'Stock prices, weather forecasting' },
            { title: 'Machine Translation', description: 'Translate between languages' }
        ],
        codeExample: `// LSTM cell concept\nforget_gate = sigmoid(Wf·[h, x] + bf)\ninput_gate = sigmoid(Wi·[h, x] + bi)\ncandidate = tanh(Wc·[h, x] + bc)\ncell = forget_gate * cell + input_gate * candidate\noutput = sigmoid(Wo·[h, x]) * tanh(cell)`,
        relatedAlgorithms: ['neural-network', 'transformer']
    },

    'transformer': {
        id: 'transformer',
        name: 'Transformer',
        fullName: 'Transformer Architecture',
        category: 'Deep Learning',
        tagline: 'Attention is all you need',
        overview: `Transformers revolutionized NLP by using self-attention mechanisms instead of recurrence. They process all positions in parallel and capture long-range dependencies efficiently.`,
        howItWorks: {
            steps: [
                'Embed input tokens with position encoding',
                'Self-attention: compute attention weights for all pairs',
                'Multi-head attention captures different relationships',
                'Feed-forward layers process each position',
                'Stack encoder/decoder blocks for deep representation'
            ],
            formula: 'Attention(Q,K,V) = softmax(QKᵀ/√d)V'
        },
        prosAndCons: {
            pros: ['Parallelizable (faster training)', 'Captures long-range dependencies', 'Foundation of GPT, BERT', 'State-of-the-art in NLP'],
            cons: ['O(n²) attention complexity', 'Requires significant compute', 'Large model sizes', 'Data hungry']
        },
        hyperparameters: [
            { name: 'd_model', description: 'Model dimension', typical: '512, 768, 1024' },
            { name: 'num_heads', description: 'Number of attention heads', typical: '8, 12, 16' },
            { name: 'num_layers', description: 'Number of encoder/decoder layers', typical: '6, 12, 24' }
        ],
        useCases: [
            { title: 'Large Language Models', description: 'GPT, ChatGPT, Claude' },
            { title: 'Machine Translation', description: 'Google Translate' },
            { title: 'Text Generation', description: 'Creative writing, code generation' }
        ],
        codeExample: `// Self-attention mechanism\nQ = X @ Wq  // Query\nK = X @ Wk  // Key  \nV = X @ Wv  // Value\nattention = softmax(Q @ K.T / sqrt(d_k)) @ V`,
        relatedAlgorithms: ['rnn-lstm', 'neural-network']
    },

    // === STATISTICS ===
    'descriptive-stats': {
        id: 'descriptive-stats',
        name: 'Mean/Median/Mode',
        fullName: 'Descriptive Statistics',
        category: 'Statistics',
        tagline: 'Measures of central tendency',
        overview: `Descriptive statistics summarize data using measures of central tendency. Mean is the average, median is the middle value, and mode is the most frequent value.`,
        howItWorks: {
            steps: [
                'Mean: Sum all values and divide by count',
                'Median: Sort data and find middle value(s)',
                'Mode: Find most frequently occurring value(s)'
            ],
            formula: 'Mean = Σxᵢ/n, Median = middle value'
        },
        prosAndCons: {
            pros: ['Easy to compute', 'Intuitive interpretation', 'Foundation for other statistics', 'Applicable to all data'],
            cons: ['Mean sensitive to outliers', 'Mode may not exist or be unique', 'Single value loses information']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Data Summary', description: 'Quickly understand data distribution' },
            { title: 'Reporting', description: 'Business metrics and KPIs' },
            { title: 'Outlier Detection', description: 'Compare mean vs median' }
        ],
        codeExample: `const mean = arr.reduce((a,b) => a+b) / arr.length;\nconst sorted = [...arr].sort((a,b) => a-b);\nconst median = sorted[Math.floor(sorted.length/2)];\nconst mode = findMostFrequent(arr);`,
        relatedAlgorithms: ['standard-deviation', 'correlation']
    },

    'standard-deviation': {
        id: 'standard-deviation',
        name: 'Standard Deviation',
        fullName: 'Standard Deviation & Variance',
        category: 'Statistics',
        tagline: 'Measuring data spread',
        overview: `Standard deviation measures the dispersion of data points from the mean. Variance is the squared standard deviation. Higher values indicate more spread.`,
        howItWorks: {
            steps: [
                'Calculate the mean of the data',
                'Subtract mean from each value (deviations)',
                'Square each deviation',
                'Average the squared deviations (variance)',
                'Take square root (standard deviation)'
            ],
            formula: 'σ = √(Σ(xᵢ - μ)² / n)'
        },
        prosAndCons: {
            pros: ['Quantifies variability', 'Same units as data', 'Foundation for z-scores', 'Used in confidence intervals'],
            cons: ['Sensitive to outliers', 'Assumes normal distribution for some applications', 'Population vs sample formula']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Risk Analysis', description: 'Measure investment volatility' },
            { title: 'Quality Control', description: 'Detect manufacturing variations' },
            { title: 'Normal Distribution', description: '68-95-99.7 rule for spread' }
        ],
        codeExample: `const mean = arr.reduce((a,b) => a+b) / arr.length;\nconst variance = arr.reduce((sum, x) => \n  sum + Math.pow(x - mean, 2), 0) / arr.length;\nconst stdDev = Math.sqrt(variance);`,
        relatedAlgorithms: ['descriptive-stats', 'correlation']
    },

    'correlation': {
        id: 'correlation',
        name: 'Correlation',
        fullName: 'Pearson Correlation Coefficient',
        category: 'Statistics',
        tagline: 'Measuring linear relationships',
        overview: `Correlation measures the strength and direction of a linear relationship between two variables. Values range from -1 (perfect negative) to +1 (perfect positive).`,
        howItWorks: {
            steps: [
                'Calculate means of both variables',
                'Compute covariance: Σ(xᵢ-x̄)(yᵢ-ȳ)',
                'Calculate standard deviations of both',
                'Divide covariance by product of std devs',
                'Result ranges from -1 to +1'
            ],
            formula: 'r = Σ(xᵢ-x̄)(yᵢ-ȳ) / (nσₓσᵧ)'
        },
        prosAndCons: {
            pros: ['Quantifies relationship strength', 'Normalized (-1 to 1)', 'Easy to interpret', 'Foundation for regression'],
            cons: ['Only measures linear relationships', 'Correlation ≠ causation', 'Sensitive to outliers', 'Requires paired data']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Feature Selection', description: 'Identify related features' },
            { title: 'Portfolio Analysis', description: 'Diversification based on correlation' },
            { title: 'Hypothesis Testing', description: 'Test for relationships' }
        ],
        codeExample: `function correlation(x, y) {\n  const mx = mean(x), my = mean(y);\n  const cov = x.reduce((s, xi, i) => \n    s + (xi - mx) * (y[i] - my), 0);\n  return cov / (std(x) * std(y) * x.length);\n}`,
        relatedAlgorithms: ['standard-deviation', 'linear-regression']
    },

    'bayesian-inference': {
        id: 'bayesian-inference',
        name: 'Bayesian Inference',
        fullName: 'Bayesian Statistical Inference',
        category: 'Statistics',
        tagline: 'Updating beliefs with evidence',
        overview: `Bayesian inference updates probability estimates as new evidence is obtained. It uses Bayes' theorem to combine prior beliefs with observed data to form posterior probabilities.`,
        howItWorks: {
            steps: [
                'Start with prior probability P(H)',
                'Observe evidence/data',
                'Calculate likelihood P(E|H)',
                'Apply Bayes theorem: P(H|E) = P(E|H)P(H)/P(E)',
                'Posterior becomes new prior for next update'
            ],
            formula: 'P(H|E) = P(E|H)P(H) / P(E)'
        },
        prosAndCons: {
            pros: ['Incorporates prior knowledge', 'Updates beliefs rationally', 'Provides probability distributions', 'Handles uncertainty well'],
            cons: ['Requires choosing prior', 'Can be computationally expensive', 'Subjective prior selection', 'Complex for high dimensions']
        },
        hyperparameters: [
            { name: 'prior', description: 'Initial probability distribution', typical: 'Uniform, Beta, Normal' }
        ],
        useCases: [
            { title: 'A/B Testing', description: 'Determine which variant is better' },
            { title: 'Medical Diagnosis', description: 'Update disease probability with tests' },
            { title: 'Spam Filtering', description: 'Update spam probability with words' }
        ],
        codeExample: `// Bayesian update\nfunction posterior(prior, likelihood, evidence) {\n  return (likelihood * prior) / evidence;\n}\n// evidence = sum over all hypotheses of P(E|H)*P(H)`,
        relatedAlgorithms: ['naive-bayes', 'markov-chain']
    },

    'markov-chain': {
        id: 'markov-chain',
        name: 'Markov Chain',
        fullName: 'Markov Chain',
        category: 'Statistics',
        tagline: 'Probabilistic state transitions',
        overview: `A Markov Chain is a stochastic model describing a sequence of events where the probability of each event depends only on the current state, not the history (memoryless property).`,
        howItWorks: {
            steps: [
                'Define state space S',
                'Define transition matrix P[i,j] = P(next=j | current=i)',
                'Start from initial state',
                'Transition based on probabilities',
                'Long-run: converges to stationary distribution'
            ],
            formula: 'P(Xₙ₊₁=j | Xₙ=i) = Pᵢⱼ'
        },
        prosAndCons: {
            pros: ['Simple probabilistic model', 'Memoryless (efficient)', 'Theoretical guarantees exist', 'Wide applications'],
            cons: ['Memoryless assumption may not hold', 'Limited to discrete states', 'May need large data for transition probabilities']
        },
        hyperparameters: [
            { name: 'states', description: 'Number of states', typical: 'Application dependent' }
        ],
        useCases: [
            { title: 'Text Generation', description: 'Generate text based on word transitions' },
            { title: 'Weather Prediction', description: 'Model weather state changes' },
            { title: 'PageRank', description: 'Web page importance via random walk' }
        ],
        codeExample: `// Markov chain step\nfunction nextState(current, transitionMatrix) {\n  const probs = transitionMatrix[current];\n  const r = Math.random();\n  let cumulative = 0;\n  for (let i = 0; i < probs.length; i++) {\n    cumulative += probs[i];\n    if (r < cumulative) return i;\n  }\n}`,
        relatedAlgorithms: ['bayesian-inference', 'pagerank']
    },

    // === SECURITY ===
    'aes': {
        id: 'aes',
        name: 'AES',
        fullName: 'Advanced Encryption Standard',
        category: 'Security',
        tagline: 'Symmetric block cipher for data encryption',
        overview: `AES is a symmetric encryption algorithm that encrypts data in fixed-size blocks (128 bits). It supports key sizes of 128, 192, or 256 bits and is widely used for secure data encryption.`,
        howItWorks: {
            steps: [
                'Key expansion: generate round keys from cipher key',
                'Initial round: AddRoundKey',
                'Main rounds: SubBytes, ShiftRows, MixColumns, AddRoundKey',
                'Final round: SubBytes, ShiftRows, AddRoundKey',
                'Number of rounds: 10/12/14 for 128/192/256-bit keys'
            ],
            formula: 'Ciphertext = AES(Plaintext, Key)'
        },
        prosAndCons: {
            pros: ['Fast and efficient', 'Strong security', 'Hardware acceleration available', 'Widely adopted standard'],
            cons: ['Key distribution challenge', 'Block cipher (needs mode of operation)', 'Key management required']
        },
        hyperparameters: [
            { name: 'key_size', description: 'Key length in bits', typical: '128, 192, or 256' },
            { name: 'mode', description: 'Mode of operation', typical: 'CBC, GCM, CTR' }
        ],
        useCases: [
            { title: 'Data Encryption', description: 'Encrypt files and databases' },
            { title: 'HTTPS/TLS', description: 'Secure web communications' },
            { title: 'Disk Encryption', description: 'Full disk encryption' }
        ],
        codeExample: `// AES encryption concept (use crypto library)\nconst cipher = crypto.createCipheriv('aes-256-gcm', key, iv);\nlet encrypted = cipher.update(plaintext, 'utf8', 'hex');\nencrypted += cipher.final('hex');`,
        relatedAlgorithms: ['rsa', 'sha-256']
    },

    'rsa': {
        id: 'rsa',
        name: 'RSA',
        fullName: 'RSA Cryptosystem',
        category: 'Security',
        tagline: 'Public-key cryptography for secure communication',
        overview: `RSA is an asymmetric cryptographic algorithm using public-private key pairs. The public key encrypts data that only the private key can decrypt, enabling secure communication without shared secrets.`,
        howItWorks: {
            steps: [
                'Generate two large primes p, q',
                'Compute n = p × q (modulus)',
                'Compute φ(n) = (p-1)(q-1)',
                'Choose e coprime to φ(n) (public exponent)',
                'Compute d = e⁻¹ mod φ(n) (private exponent)',
                'Encrypt: c = mᵉ mod n, Decrypt: m = cᵈ mod n'
            ],
            formula: 'c = mᵉ mod n, m = cᵈ mod n'
        },
        prosAndCons: {
            pros: ['No shared secret needed', 'Enables digital signatures', 'Well-studied security', 'Widely implemented'],
            cons: ['Slow for large data', 'Large key sizes needed', 'Vulnerable to quantum computers', 'Key generation complex']
        },
        hyperparameters: [
            { name: 'key_size', description: 'Key length in bits', typical: '2048, 4096' },
            { name: 'exponent', description: 'Public exponent e', typical: '65537' }
        ],
        useCases: [
            { title: 'Key Exchange', description: 'Securely share symmetric keys' },
            { title: 'Digital Signatures', description: 'Sign documents and code' },
            { title: 'TLS/SSL', description: 'Certificate-based authentication' }
        ],
        codeExample: `// RSA key pair generation concept\n// n = p * q (large primes)\n// e: public exponent (65537)\n// d: private exponent (e^-1 mod φ(n))\n// Encrypt: cipher = message^e mod n\n// Decrypt: message = cipher^d mod n`,
        relatedAlgorithms: ['aes', 'diffie-hellman']
    },

    'sha-256': {
        id: 'sha-256',
        name: 'SHA-256',
        fullName: 'Secure Hash Algorithm 256-bit',
        category: 'Security',
        tagline: 'Cryptographic hash for data integrity',
        overview: `SHA-256 is a cryptographic hash function that produces a 256-bit (32-byte) hash value. It's deterministic, one-way, and collision-resistant, making it ideal for data integrity verification.`,
        howItWorks: {
            steps: [
                'Pad message to multiple of 512 bits',
                'Parse into 512-bit blocks',
                'Initialize hash values (first 32 bits of fractional parts of sqrt of first 8 primes)',
                'For each block: expand, compress with rounds',
                'Add compressed chunk to current hash value',
                'Output 256-bit hash'
            ],
            formula: 'H(m) = 256-bit digest'
        },
        prosAndCons: {
            pros: ['One-way function', 'Collision resistant', 'Fixed output size', 'Fast computation'],
            cons: ['Not reversible', 'Vulnerable to length extension', 'No built-in salt', 'Use bcrypt for passwords']
        },
        hyperparameters: [],
        useCases: [
            { title: 'Data Integrity', description: 'Verify file/message integrity' },
            { title: 'Blockchain', description: 'Bitcoin mining and block validation' },
            { title: 'Digital Signatures', description: 'Hash before signing' }
        ],
        codeExample: `// SHA-256 hashing\nconst hash = crypto.createHash('sha256');\nhash.update(data);\nconst digest = hash.digest('hex');\n// Output: 64 hex characters (256 bits)`,
        relatedAlgorithms: ['hmac', 'aes']
    },

    'hmac': {
        id: 'hmac',
        name: 'HMAC',
        fullName: 'Hash-based Message Authentication Code',
        category: 'Security',
        tagline: 'Message authentication with secret key',
        overview: `HMAC combines a cryptographic hash function with a secret key to provide both data integrity and authentication. It verifies that a message hasn't been altered and comes from the expected sender.`,
        howItWorks: {
            steps: [
                'If key longer than block size, hash it',
                'Pad key to block size',
                'XOR key with inner and outer pads',
                'HMAC = H((K ⊕ opad) || H((K ⊕ ipad) || message))',
                'Compare received and computed HMAC'
            ],
            formula: 'HMAC(K, m) = H((K ⊕ opad) || H((K ⊕ ipad) || m))'
        },
        prosAndCons: {
            pros: ['Provides authentication', 'Resistant to length extension', 'Uses standard hash functions', 'Fast computation'],
            cons: ['Requires shared secret', 'Key management needed', 'Not encryption']
        },
        hyperparameters: [
            { name: 'hash', description: 'Underlying hash function', typical: 'SHA-256, SHA-384' }
        ],
        useCases: [
            { title: 'API Authentication', description: 'Sign API requests' },
            { title: 'JWT Tokens', description: 'Verify token integrity' },
            { title: 'Message Verification', description: 'Ensure message authenticity' }
        ],
        codeExample: `// HMAC-SHA256\nconst hmac = crypto.createHmac('sha256', secretKey);\nhmac.update(message);\nconst signature = hmac.digest('hex');`,
        relatedAlgorithms: ['sha-256', 'aes']
    },

    'diffie-hellman': {
        id: 'diffie-hellman',
        name: 'Diffie-Hellman',
        fullName: 'Diffie-Hellman Key Exchange',
        category: 'Security',
        tagline: 'Secure key exchange over public channel',
        overview: `Diffie-Hellman enables two parties to establish a shared secret over an insecure channel. The security is based on the difficulty of the discrete logarithm problem.`,
        howItWorks: {
            steps: [
                'Agree on public parameters: prime p, generator g',
                'Alice picks private a, computes A = gᵃ mod p',
                'Bob picks private b, computes B = gᵇ mod p',
                'Exchange A and B publicly',
                'Alice computes s = Bᵃ mod p, Bob computes s = Aᵇ mod p',
                'Both have shared secret s = gᵃᵇ mod p'
            ],
            formula: 's = gᵃᵇ mod p'
        },
        prosAndCons: {
            pros: ['No prior shared secret needed', 'Forward secrecy possible', 'Foundation of key exchange', 'Well-studied'],
            cons: ['Vulnerable to MITM without authentication', 'Discrete log must be hard', 'Parameter generation important']
        },
        hyperparameters: [
            { name: 'prime_size', description: 'Size of prime p', typical: '2048+ bits' },
            { name: 'generator', description: 'Generator of group', typical: '2 or 5' }
        ],
        useCases: [
            { title: 'TLS/SSL', description: 'Establish session keys' },
            { title: 'VPN', description: 'Secure tunnel establishment' },
            { title: 'Secure Messaging', description: 'Signal protocol' }
        ],
        codeExample: `// Diffie-Hellman concept\nconst alice = crypto.createDiffieHellman(2048);\nalice.generateKeys();\nconst alicePublic = alice.getPublicKey();\n// Exchange public keys, compute shared secret\nconst secret = alice.computeSecret(bobPublic);`,
        relatedAlgorithms: ['rsa', 'aes']
    },

    // === RECOMMENDATION ===
    'collaborative-filtering': {
        id: 'collaborative-filtering',
        name: 'Collaborative Filtering',
        fullName: 'Collaborative Filtering Recommendation',
        category: 'Recommendation',
        tagline: 'Recommend based on similar users or items',
        overview: `Collaborative Filtering recommends items based on collective user behavior. User-based CF finds similar users; item-based CF finds similar items. Matrix factorization is a modern approach.`,
        howItWorks: {
            steps: [
                'Build user-item interaction matrix',
                'User-based: find users with similar ratings',
                'Item-based: find items rated similarly',
                'Matrix factorization: decompose into user/item embeddings',
                'Predict missing ratings, recommend top items'
            ],
            formula: 'R̂ ≈ U × Vᵀ (matrix factorization)'
        },
        prosAndCons: {
            pros: ['No content features needed', 'Discovers unexpected recommendations', 'Improves with more users', 'Captures complex preferences'],
            cons: ['Cold start problem', 'Sparsity issues', 'Scalability challenges', 'Popularity bias']
        },
        hyperparameters: [
            { name: 'k', description: 'Number of neighbors', typical: '10-50' },
            { name: 'latent_factors', description: 'Embedding dimensions', typical: '50-200' }
        ],
        useCases: [
            { title: 'Movie Recommendations', description: 'Netflix, IMDb' },
            { title: 'Music Playlists', description: 'Spotify Discover Weekly' },
            { title: 'E-commerce', description: 'Amazon product recommendations' }
        ],
        codeExample: `// User-based collaborative filtering\nfunction recommend(user, ratings, k=10) {\n  const similar = findSimilarUsers(user, ratings, k);\n  const candidates = getUnratedItems(user, similar);\n  return rankByPredictedRating(candidates, similar);\n}`,
        relatedAlgorithms: ['content-based-filtering', 'knn']
    },

    'content-based-filtering': {
        id: 'content-based-filtering',
        name: 'Content-Based Filtering',
        fullName: 'Content-Based Recommendation',
        category: 'Recommendation',
        tagline: 'Recommend based on item features',
        overview: `Content-Based Filtering recommends items similar to those a user has liked, based on item features. It builds a user profile from liked item features and matches against candidate items.`,
        howItWorks: {
            steps: [
                'Extract features from items (genre, keywords, etc.)',
                'Build user profile from liked item features',
                'TF-IDF or embeddings for text features',
                'Compute similarity between profile and candidates',
                'Recommend highest similarity items'
            ],
            formula: 'score = cos(userProfile, itemFeatures)'
        },
        prosAndCons: {
            pros: ['No cold start for items', 'Transparent recommendations', 'User independence', 'Works with new items'],
            cons: ['Overspecialization', 'Cold start for users', 'Requires good features', 'Limited discovery']
        },
        hyperparameters: [
            { name: 'similarity', description: 'Similarity metric', typical: 'Cosine, Jaccard' },
            { name: 'features', description: 'Feature extraction method', typical: 'TF-IDF, embeddings' }
        ],
        useCases: [
            { title: 'News Recommendations', description: 'Articles similar to read history' },
            { title: 'Product Recommendations', description: 'Items with similar descriptions' },
            { title: 'Music Discovery', description: 'Songs with similar audio features' }
        ],
        codeExample: `// Content-based filtering\nfunction recommend(userProfile, items) {\n  return items\n    .map(item => ({\n      item,\n      score: cosineSimilarity(userProfile, item.features)\n    }))\n    .sort((a, b) => b.score - a.score);\n}`,
        relatedAlgorithms: ['collaborative-filtering', 'tf-idf-nb']
    },

    'lru-cache': {
        id: 'lru-cache',
        name: 'LRU Cache',
        fullName: 'Least Recently Used Cache',
        category: 'Recommendation',
        tagline: 'Efficient caching with eviction policy',
        overview: `LRU Cache evicts the least recently used items when capacity is reached. It provides O(1) access and updates using a hash map combined with a doubly linked list.`,
        howItWorks: {
            steps: [
                'Maintain hash map for O(1) lookup',
                'Maintain doubly linked list for recency order',
                'On access: move item to front of list',
                'On insert at capacity: remove tail (LRU)',
                'Update hash map accordingly'
            ],
            formula: 'O(1) get and put operations'
        },
        prosAndCons: {
            pros: ['O(1) operations', 'Simple eviction policy', 'Good for temporal locality', 'Wide applicability'],
            cons: ['Fixed capacity', 'No frequency consideration (see LFU)', 'Memory overhead', 'Not optimal for all access patterns']
        },
        hyperparameters: [
            { name: 'capacity', description: 'Maximum cache size', typical: 'Application dependent' }
        ],
        useCases: [
            { title: 'Web Caching', description: 'Cache recently accessed pages' },
            { title: 'Database Caching', description: 'Buffer pool management' },
            { title: 'API Rate Limiting', description: 'Track recent requests' }
        ],
        codeExample: `class LRUCache {\n  constructor(capacity) {\n    this.capacity = capacity;\n    this.cache = new Map();\n  }\n  get(key) {\n    if (!this.cache.has(key)) return -1;\n    const val = this.cache.get(key);\n    this.cache.delete(key);\n    this.cache.set(key, val);\n    return val;\n  }\n}`,
        relatedAlgorithms: ['collaborative-filtering']
    }
};
