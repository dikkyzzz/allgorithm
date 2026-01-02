export interface Algorithm {
  id: string;
  name: string;
  category: 'Clustering' | 'Classification' | 'Regression' | 'Dimensionality Reduction' | 'Text Mining' | 'Sorting' | 'Searching' | 'Graph' | 'Deep Learning' | 'Statistics' | 'Security' | 'Recommendation';
  description: string;
  bestFor: string;
  requirements: string[];
}

export const algorithms: Algorithm[] = [
  // === CLUSTERING ===
  {
    id: 'k-means',
    name: 'K-Means',
    category: 'Clustering',
    description: 'Partitions data into K distinct clusters based on feature similarity.',
    bestFor: 'Customer segmentation, pattern recognition.',
    requirements: ['Numerical data', 'Specified K (clusters)'],
  },
  {
    id: 'dbscan',
    name: 'DBSCAN',
    category: 'Clustering',
    description: 'Density-based spatial clustering of applications with noise.',
    bestFor: 'Anomalous detection, non-spherical clusters.',
    requirements: ['Spatial data', 'Epsilon & MinPts'],
  },
  {
    id: 'hierarchical-clustering',
    name: 'Hierarchical Clustering',
    category: 'Clustering',
    description: 'Builds a tree of clusters by iteratively merging or splitting groups.',
    bestFor: 'Dendrogram visualization, taxonomy creation.',
    requirements: ['Distance metric', 'Linkage method'],
  },

  // === CLASSIFICATION ===
  {
    id: 'naive-bayes',
    name: 'Naive Bayes',
    category: 'Classification',
    description: 'Probabilistic classifier based on applying Bayes theorem with strong independence assumptions.',
    bestFor: 'Spam detection, sentiment analysis.',
    requirements: ['Categorical/Numerical data', 'Labeled dataset'],
  },
  {
    id: 'logistic-regression',
    name: 'Logistic Regression',
    category: 'Classification',
    description: 'Predicts the probability of a binary outcome.',
    bestFor: 'Churn prediction, credit scoring.',
    requirements: ['Independent variables', 'Binary target'],
  },
  {
    id: 'svm',
    name: 'SVM',
    category: 'Classification',
    description: 'Supervised learning models that analyze data for classification and regression analysis.',
    bestFor: 'Image recognition, bioinformatics.',
    requirements: ['Clear margin of separation', 'High-dimensional space'],
  },
  {
    id: 'decision-tree',
    name: 'Decision Tree',
    category: 'Classification',
    description: 'Tree-like model that makes decisions based on feature values at each node.',
    bestFor: 'Rule-based classification, interpretable models.',
    requirements: ['Categorical/Numerical data', 'Labeled dataset'],
  },
  {
    id: 'random-forest',
    name: 'Random Forest',
    category: 'Classification',
    description: 'Ensemble of decision trees that improves accuracy through bagging and feature randomization.',
    bestFor: 'Complex classification, feature importance analysis.',
    requirements: ['Large dataset', 'Labeled data'],
  },
  {
    id: 'knn',
    name: 'KNN',
    category: 'Classification',
    description: 'Classifies data points based on the majority class of their k nearest neighbors.',
    bestFor: 'Pattern recognition, recommendation systems.',
    requirements: ['Numerical data', 'Specified K (neighbors)'],
  },

  // === REGRESSION ===
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    category: 'Regression',
    description: 'Models the relationship between a dependent variable and one or more independent variables.',
    bestFor: 'Sales forecasting, trend analysis.',
    requirements: ['Continuous data', 'Linear relationship'],
  },

  // === DIMENSIONALITY REDUCTION ===
  {
    id: 'pca',
    name: 'PCA',
    category: 'Dimensionality Reduction',
    description: 'Reduces the dimensionality of data while preserving as much variance as possible.',
    bestFor: 'Data visualization, noise reduction.',
    requirements: ['Numerical data', 'High dimensionality'],
  },

  // === TEXT MINING ===
  {
    id: 'tf-idf-nb',
    name: 'TF-IDF + Naive Bayes',
    category: 'Text Mining',
    description: 'Combines term frequency-inverse document frequency with Naive Bayes for text classification.',
    bestFor: 'Document categorization, topic discovery.',
    requirements: ['Text/Document data', 'Predefined labels'],
  },

  // === SORTING ===
  {
    id: 'quick-sort',
    name: 'Quick Sort',
    category: 'Sorting',
    description: 'Efficient divide-and-conquer sorting algorithm using pivot partitioning.',
    bestFor: 'General-purpose sorting, large datasets.',
    requirements: ['Array/List data'],
  },
  {
    id: 'merge-sort',
    name: 'Merge Sort',
    category: 'Sorting',
    description: 'Stable divide-and-conquer sorting algorithm that merges sorted subarrays.',
    bestFor: 'Stable sorting, linked lists.',
    requirements: ['Array/List data'],
  },
  {
    id: 'bubble-sort',
    name: 'Bubble Sort',
    category: 'Sorting',
    description: 'Simple comparison-based sorting that repeatedly swaps adjacent elements.',
    bestFor: 'Educational purposes, small datasets.',
    requirements: ['Array/List data'],
  },
  {
    id: 'heap-sort',
    name: 'Heap Sort',
    category: 'Sorting',
    description: 'Comparison-based sorting using a binary heap data structure.',
    bestFor: 'Guaranteed O(n log n), in-place sorting.',
    requirements: ['Array/List data'],
  },
  {
    id: 'insertion-sort',
    name: 'Insertion Sort',
    category: 'Sorting',
    description: 'Simple sorting that builds the sorted array one element at a time.',
    bestFor: 'Small datasets, nearly sorted data.',
    requirements: ['Array/List data'],
  },

  // === SEARCHING ===
  {
    id: 'binary-search',
    name: 'Binary Search',
    category: 'Searching',
    description: 'Efficiently finds elements in a sorted array by repeatedly dividing the search interval.',
    bestFor: 'Fast lookup in sorted data.',
    requirements: ['Sorted array', 'Target value'],
  },
  {
    id: 'linear-search',
    name: 'Linear Search',
    category: 'Searching',
    description: 'Sequentially checks each element until a match is found.',
    bestFor: 'Unsorted data, small datasets.',
    requirements: ['Array/List data', 'Target value'],
  },

  // === GRAPH ===
  {
    id: 'dijkstra',
    name: 'Dijkstra',
    category: 'Graph',
    description: 'Finds the shortest path from a source node to all other nodes in a weighted graph.',
    bestFor: 'Route planning, network optimization.',
    requirements: ['Weighted graph', 'Non-negative weights'],
  },
  {
    id: 'bfs-dfs',
    name: 'BFS/DFS',
    category: 'Graph',
    description: 'Graph traversal algorithms: Breadth-First Search explores level by level, Depth-First Search explores as deep as possible.',
    bestFor: 'Path finding, connected components.',
    requirements: ['Graph structure'],
  },
  {
    id: 'a-star',
    name: 'A*',
    category: 'Graph',
    description: 'Heuristic search algorithm that finds the shortest path using estimated cost to goal.',
    bestFor: 'Game pathfinding, navigation systems.',
    requirements: ['Weighted graph', 'Heuristic function'],
  },
  {
    id: 'bellman-ford',
    name: 'Bellman-Ford',
    category: 'Graph',
    description: 'Finds shortest paths from a source vertex, handling negative edge weights.',
    bestFor: 'Graphs with negative weights, detecting negative cycles.',
    requirements: ['Weighted graph'],
  },
  {
    id: 'floyd-warshall',
    name: 'Floyd-Warshall',
    category: 'Graph',
    description: 'Finds shortest paths between all pairs of vertices in a weighted graph.',
    bestFor: 'All-pairs shortest path, dense graphs.',
    requirements: ['Weighted graph', 'Adjacency matrix'],
  },
  {
    id: 'pagerank',
    name: 'PageRank',
    category: 'Graph',
    description: 'Ranks nodes in a graph based on the structure of incoming links.',
    bestFor: 'Web page ranking, influence analysis.',
    requirements: ['Directed graph', 'Damping factor'],
  },

  // === DEEP LEARNING ===
  {
    id: 'neural-network',
    name: 'Neural Network',
    category: 'Deep Learning',
    description: 'Artificial neural network with interconnected layers of neurons for learning complex patterns.',
    bestFor: 'Pattern recognition, function approximation.',
    requirements: ['Large dataset', 'GPU recommended'],
  },
  {
    id: 'cnn',
    name: 'CNN',
    category: 'Deep Learning',
    description: 'Convolutional Neural Network designed for processing grid-like data such as images.',
    bestFor: 'Image classification, object detection.',
    requirements: ['Image data', 'GPU required'],
  },
  {
    id: 'rnn-lstm',
    name: 'RNN/LSTM',
    category: 'Deep Learning',
    description: 'Recurrent networks with memory cells for processing sequential data.',
    bestFor: 'Time series, natural language processing.',
    requirements: ['Sequential data', 'GPU recommended'],
  },
  {
    id: 'transformer',
    name: 'Transformer',
    category: 'Deep Learning',
    description: 'Attention-based architecture that processes sequences in parallel.',
    bestFor: 'NLP, large language models (GPT, BERT).',
    requirements: ['Large dataset', 'Significant compute'],
  },

  // === STATISTICS ===
  {
    id: 'descriptive-stats',
    name: 'Mean/Median/Mode',
    category: 'Statistics',
    description: 'Measures of central tendency that summarize data distribution.',
    bestFor: 'Data summarization, understanding distributions.',
    requirements: ['Numerical data'],
  },
  {
    id: 'standard-deviation',
    name: 'Standard Deviation',
    category: 'Statistics',
    description: 'Measures the amount of variation or dispersion in a dataset.',
    bestFor: 'Risk analysis, quality control.',
    requirements: ['Numerical data'],
  },
  {
    id: 'correlation',
    name: 'Correlation',
    category: 'Statistics',
    description: 'Measures the statistical relationship between two variables.',
    bestFor: 'Feature selection, relationship analysis.',
    requirements: ['Paired numerical data'],
  },
  {
    id: 'bayesian-inference',
    name: 'Bayesian Inference',
    category: 'Statistics',
    description: 'Updates probability estimates as new evidence is obtained using Bayes theorem.',
    bestFor: 'Probabilistic reasoning, A/B testing.',
    requirements: ['Prior distribution', 'Likelihood function'],
  },
  {
    id: 'markov-chain',
    name: 'Markov Chain',
    category: 'Statistics',
    description: 'Stochastic model describing a sequence of events where probability depends only on current state.',
    bestFor: 'Text generation, weather prediction.',
    requirements: ['State space', 'Transition probabilities'],
  },

  // === SECURITY ===
  {
    id: 'aes',
    name: 'AES',
    category: 'Security',
    description: 'Advanced Encryption Standard - symmetric block cipher for data encryption.',
    bestFor: 'Data encryption, secure communication.',
    requirements: ['Key (128/192/256 bit)', 'Plaintext'],
  },
  {
    id: 'rsa',
    name: 'RSA',
    category: 'Security',
    description: 'Asymmetric cryptographic algorithm using public and private key pairs.',
    bestFor: 'Secure key exchange, digital signatures.',
    requirements: ['Public/Private key pair'],
  },
  {
    id: 'sha-256',
    name: 'SHA-256',
    category: 'Security',
    description: 'Cryptographic hash function producing a 256-bit hash value.',
    bestFor: 'Data integrity, password hashing.',
    requirements: ['Input data'],
  },
  {
    id: 'hmac',
    name: 'HMAC',
    category: 'Security',
    description: 'Hash-based Message Authentication Code for message integrity and authenticity.',
    bestFor: 'API authentication, message verification.',
    requirements: ['Secret key', 'Message'],
  },
  {
    id: 'diffie-hellman',
    name: 'Diffie-Hellman',
    category: 'Security',
    description: 'Key exchange protocol for securely sharing cryptographic keys over public channel.',
    bestFor: 'Secure key exchange, TLS/SSL.',
    requirements: ['Prime number', 'Generator'],
  },

  // === RECOMMENDATION ===
  {
    id: 'collaborative-filtering',
    name: 'Collaborative Filtering',
    category: 'Recommendation',
    description: 'Recommends items based on preferences of similar users or items.',
    bestFor: 'Movie/music recommendations, e-commerce.',
    requirements: ['User-item interaction matrix'],
  },
  {
    id: 'content-based-filtering',
    name: 'Content-Based Filtering',
    category: 'Recommendation',
    description: 'Recommends items similar to those a user has liked based on item features.',
    bestFor: 'News recommendations, personalized content.',
    requirements: ['Item features', 'User preferences'],
  },
  {
    id: 'lru-cache',
    name: 'LRU Cache',
    category: 'Recommendation',
    description: 'Least Recently Used cache eviction algorithm for efficient data access.',
    bestFor: 'Caching, memory management.',
    requirements: ['Cache capacity', 'Access pattern'],
  },
];
