// Sample datasets for ALLgorithm
// Built-in datasets that users can use without uploading

export interface DatasetInfo {
    id: string;
    name: string;
    description: string;
    rows: number;
    features: string[];
    targetColumn?: string;
    type: 'clustering' | 'classification' | 'regression';
}

export interface Dataset extends DatasetInfo {
    data: Record<string, number | string>[];
}

// Iris Dataset (for clustering and classification)
const irisData = [
    { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 4.7, sepal_width: 3.2, petal_length: 1.3, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 4.6, sepal_width: 3.1, petal_length: 1.5, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 5.0, sepal_width: 3.6, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 5.4, sepal_width: 3.9, petal_length: 1.7, petal_width: 0.4, species: 'setosa' },
    { sepal_length: 4.6, sepal_width: 3.4, petal_length: 1.4, petal_width: 0.3, species: 'setosa' },
    { sepal_length: 5.0, sepal_width: 3.4, petal_length: 1.5, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 4.4, sepal_width: 2.9, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 4.9, sepal_width: 3.1, petal_length: 1.5, petal_width: 0.1, species: 'setosa' },
    { sepal_length: 5.4, sepal_width: 3.7, petal_length: 1.5, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 4.8, sepal_width: 3.4, petal_length: 1.6, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 4.8, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.1, species: 'setosa' },
    { sepal_length: 4.3, sepal_width: 3.0, petal_length: 1.1, petal_width: 0.1, species: 'setosa' },
    { sepal_length: 5.8, sepal_width: 4.0, petal_length: 1.2, petal_width: 0.2, species: 'setosa' },
    { sepal_length: 5.7, sepal_width: 4.4, petal_length: 1.5, petal_width: 0.4, species: 'setosa' },
    { sepal_length: 5.4, sepal_width: 3.9, petal_length: 1.3, petal_width: 0.4, species: 'setosa' },
    { sepal_length: 7.0, sepal_width: 3.2, petal_length: 4.7, petal_width: 1.4, species: 'versicolor' },
    { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5, species: 'versicolor' },
    { sepal_length: 6.9, sepal_width: 3.1, petal_length: 4.9, petal_width: 1.5, species: 'versicolor' },
    { sepal_length: 5.5, sepal_width: 2.3, petal_length: 4.0, petal_width: 1.3, species: 'versicolor' },
    { sepal_length: 6.5, sepal_width: 2.8, petal_length: 4.6, petal_width: 1.5, species: 'versicolor' },
    { sepal_length: 5.7, sepal_width: 2.8, petal_length: 4.5, petal_width: 1.3, species: 'versicolor' },
    { sepal_length: 6.3, sepal_width: 3.3, petal_length: 4.7, petal_width: 1.6, species: 'versicolor' },
    { sepal_length: 4.9, sepal_width: 2.4, petal_length: 3.3, petal_width: 1.0, species: 'versicolor' },
    { sepal_length: 6.6, sepal_width: 2.9, petal_length: 4.6, petal_width: 1.3, species: 'versicolor' },
    { sepal_length: 5.2, sepal_width: 2.7, petal_length: 3.9, petal_width: 1.4, species: 'versicolor' },
    { sepal_length: 5.0, sepal_width: 2.0, petal_length: 3.5, petal_width: 1.0, species: 'versicolor' },
    { sepal_length: 5.9, sepal_width: 3.0, petal_length: 4.2, petal_width: 1.5, species: 'versicolor' },
    { sepal_length: 6.0, sepal_width: 2.2, petal_length: 4.0, petal_width: 1.0, species: 'versicolor' },
    { sepal_length: 6.1, sepal_width: 2.9, petal_length: 4.7, petal_width: 1.4, species: 'versicolor' },
    { sepal_length: 5.6, sepal_width: 2.9, petal_length: 3.6, petal_width: 1.3, species: 'versicolor' },
    { sepal_length: 6.7, sepal_width: 3.1, petal_length: 4.4, petal_width: 1.4, species: 'versicolor' },
    { sepal_length: 5.6, sepal_width: 3.0, petal_length: 4.5, petal_width: 1.5, species: 'versicolor' },
    { sepal_length: 6.3, sepal_width: 2.5, petal_length: 4.9, petal_width: 1.5, species: 'virginica' },
    { sepal_length: 6.5, sepal_width: 3.0, petal_length: 5.2, petal_width: 2.0, species: 'virginica' },
    { sepal_length: 6.2, sepal_width: 3.4, petal_length: 5.4, petal_width: 2.3, species: 'virginica' },
    { sepal_length: 5.9, sepal_width: 3.0, petal_length: 5.1, petal_width: 1.8, species: 'virginica' },
    { sepal_length: 6.3, sepal_width: 2.9, petal_length: 5.6, petal_width: 1.8, species: 'virginica' },
    { sepal_length: 6.1, sepal_width: 2.6, petal_length: 5.6, petal_width: 1.4, species: 'virginica' },
    { sepal_length: 7.7, sepal_width: 3.0, petal_length: 6.1, petal_width: 2.3, species: 'virginica' },
    { sepal_length: 6.3, sepal_width: 3.4, petal_length: 5.6, petal_width: 2.4, species: 'virginica' },
    { sepal_length: 6.4, sepal_width: 3.1, petal_length: 5.5, petal_width: 1.8, species: 'virginica' },
    { sepal_length: 6.0, sepal_width: 3.0, petal_length: 4.8, petal_width: 1.8, species: 'virginica' },
    { sepal_length: 6.9, sepal_width: 3.1, petal_length: 5.4, petal_width: 2.1, species: 'virginica' },
    { sepal_length: 6.7, sepal_width: 3.1, petal_length: 5.6, petal_width: 2.4, species: 'virginica' },
    { sepal_length: 6.9, sepal_width: 3.2, petal_length: 5.7, petal_width: 2.3, species: 'virginica' },
    { sepal_length: 5.8, sepal_width: 2.7, petal_length: 5.1, petal_width: 1.9, species: 'virginica' },
    { sepal_length: 6.8, sepal_width: 3.2, petal_length: 5.9, petal_width: 2.3, species: 'virginica' },
    { sepal_length: 6.7, sepal_width: 3.3, petal_length: 5.7, petal_width: 2.5, species: 'virginica' },
];

// Customer Segmentation Dataset (for clustering)
const customerData = [
    { customer_id: 1, age: 19, annual_income: 15, spending_score: 39 },
    { customer_id: 2, age: 21, annual_income: 15, spending_score: 81 },
    { customer_id: 3, age: 20, annual_income: 16, spending_score: 6 },
    { customer_id: 4, age: 23, annual_income: 16, spending_score: 77 },
    { customer_id: 5, age: 31, annual_income: 17, spending_score: 40 },
    { customer_id: 6, age: 22, annual_income: 17, spending_score: 76 },
    { customer_id: 7, age: 35, annual_income: 18, spending_score: 6 },
    { customer_id: 8, age: 23, annual_income: 18, spending_score: 94 },
    { customer_id: 9, age: 64, annual_income: 19, spending_score: 3 },
    { customer_id: 10, age: 30, annual_income: 19, spending_score: 72 },
    { customer_id: 11, age: 67, annual_income: 19, spending_score: 14 },
    { customer_id: 12, age: 35, annual_income: 20, spending_score: 99 },
    { customer_id: 13, age: 58, annual_income: 20, spending_score: 15 },
    { customer_id: 14, age: 24, annual_income: 20, spending_score: 77 },
    { customer_id: 15, age: 37, annual_income: 21, spending_score: 13 },
    { customer_id: 16, age: 22, annual_income: 21, spending_score: 79 },
    { customer_id: 17, age: 35, annual_income: 23, spending_score: 35 },
    { customer_id: 18, age: 20, annual_income: 23, spending_score: 66 },
    { customer_id: 19, age: 52, annual_income: 24, spending_score: 29 },
    { customer_id: 20, age: 35, annual_income: 24, spending_score: 98 },
    { customer_id: 21, age: 35, annual_income: 25, spending_score: 35 },
    { customer_id: 22, age: 25, annual_income: 25, spending_score: 73 },
    { customer_id: 23, age: 46, annual_income: 26, spending_score: 5 },
    { customer_id: 24, age: 31, annual_income: 26, spending_score: 82 },
    { customer_id: 25, age: 54, annual_income: 28, spending_score: 33 },
    { customer_id: 26, age: 29, annual_income: 28, spending_score: 87 },
    { customer_id: 27, age: 45, annual_income: 32, spending_score: 15 },
    { customer_id: 28, age: 34, annual_income: 33, spending_score: 69 },
    { customer_id: 29, age: 40, annual_income: 33, spending_score: 17 },
    { customer_id: 30, age: 23, annual_income: 33, spending_score: 73 },
    { customer_id: 31, age: 48, annual_income: 39, spending_score: 91 },
    { customer_id: 32, age: 33, annual_income: 54, spending_score: 14 },
    { customer_id: 33, age: 50, annual_income: 54, spending_score: 47 },
    { customer_id: 34, age: 27, annual_income: 54, spending_score: 54 },
    { customer_id: 35, age: 55, annual_income: 62, spending_score: 42 },
    { customer_id: 36, age: 38, annual_income: 63, spending_score: 50 },
    { customer_id: 37, age: 35, annual_income: 72, spending_score: 35 },
    { customer_id: 38, age: 32, annual_income: 73, spending_score: 40 },
    { customer_id: 39, age: 42, annual_income: 78, spending_score: 17 },
    { customer_id: 40, age: 36, annual_income: 78, spending_score: 91 },
];

// Titanic Dataset (simplified for classification)
const titanicData = [
    { pclass: 3, sex: 1, age: 22, sibsp: 1, parch: 0, fare: 7.25, survived: 0 },
    { pclass: 1, sex: 0, age: 38, sibsp: 1, parch: 0, fare: 71.28, survived: 1 },
    { pclass: 3, sex: 0, age: 26, sibsp: 0, parch: 0, fare: 7.93, survived: 1 },
    { pclass: 1, sex: 0, age: 35, sibsp: 1, parch: 0, fare: 53.1, survived: 1 },
    { pclass: 3, sex: 1, age: 35, sibsp: 0, parch: 0, fare: 8.05, survived: 0 },
    { pclass: 3, sex: 1, age: 28, sibsp: 0, parch: 0, fare: 8.46, survived: 0 },
    { pclass: 1, sex: 1, age: 54, sibsp: 0, parch: 0, fare: 51.86, survived: 0 },
    { pclass: 3, sex: 1, age: 2, sibsp: 3, parch: 1, fare: 21.08, survived: 0 },
    { pclass: 3, sex: 0, age: 27, sibsp: 0, parch: 2, fare: 11.13, survived: 1 },
    { pclass: 2, sex: 0, age: 14, sibsp: 1, parch: 0, fare: 30.07, survived: 1 },
    { pclass: 3, sex: 0, age: 4, sibsp: 1, parch: 1, fare: 16.7, survived: 1 },
    { pclass: 1, sex: 0, age: 58, sibsp: 0, parch: 0, fare: 26.55, survived: 1 },
    { pclass: 3, sex: 1, age: 20, sibsp: 0, parch: 0, fare: 8.05, survived: 0 },
    { pclass: 3, sex: 1, age: 39, sibsp: 1, parch: 5, fare: 31.28, survived: 0 },
    { pclass: 3, sex: 0, age: 14, sibsp: 0, parch: 0, fare: 7.85, survived: 0 },
    { pclass: 2, sex: 0, age: 55, sibsp: 0, parch: 0, fare: 16, survived: 1 },
    { pclass: 3, sex: 1, age: 2, sibsp: 4, parch: 1, fare: 29.13, survived: 0 },
    { pclass: 2, sex: 1, age: 28, sibsp: 0, parch: 0, fare: 13, survived: 1 },
    { pclass: 3, sex: 0, age: 31, sibsp: 1, parch: 0, fare: 18, survived: 0 },
    { pclass: 3, sex: 0, age: 28, sibsp: 0, parch: 0, fare: 7.23, survived: 1 },
    { pclass: 2, sex: 1, age: 35, sibsp: 0, parch: 0, fare: 26, survived: 0 },
    { pclass: 2, sex: 1, age: 34, sibsp: 0, parch: 0, fare: 13, survived: 1 },
    { pclass: 3, sex: 0, age: 15, sibsp: 0, parch: 0, fare: 8.03, survived: 1 },
    { pclass: 1, sex: 1, age: 28, sibsp: 0, parch: 0, fare: 35.5, survived: 1 },
    { pclass: 3, sex: 0, age: 8, sibsp: 3, parch: 1, fare: 21.08, survived: 0 },
    { pclass: 3, sex: 0, age: 38, sibsp: 1, parch: 5, fare: 31.39, survived: 1 },
    { pclass: 3, sex: 1, age: 28, sibsp: 0, parch: 0, fare: 7.9, survived: 0 },
    { pclass: 1, sex: 1, age: 19, sibsp: 3, parch: 2, fare: 263, survived: 0 },
    { pclass: 3, sex: 0, age: 28, sibsp: 0, parch: 0, fare: 7.88, survived: 1 },
    { pclass: 3, sex: 1, age: 28, sibsp: 0, parch: 0, fare: 8.66, survived: 0 },
    { pclass: 1, sex: 1, age: 40, sibsp: 0, parch: 0, fare: 27.72, survived: 0 },
    { pclass: 2, sex: 0, age: 28, sibsp: 1, parch: 0, fare: 21, survived: 1 },
    { pclass: 1, sex: 0, age: 28, sibsp: 0, parch: 0, fare: 211.34, survived: 1 },
    { pclass: 3, sex: 1, age: 66, sibsp: 0, parch: 0, fare: 10.5, survived: 0 },
    { pclass: 3, sex: 1, age: 28, sibsp: 1, parch: 0, fare: 14.5, survived: 0 },
    { pclass: 1, sex: 1, age: 42, sibsp: 1, parch: 0, fare: 52, survived: 0 },
    { pclass: 3, sex: 1, age: 28, sibsp: 2, parch: 0, fare: 21.68, survived: 0 },
    { pclass: 2, sex: 1, age: 21, sibsp: 0, parch: 0, fare: 73.5, survived: 0 },
    { pclass: 3, sex: 0, age: 18, sibsp: 2, parch: 0, fare: 18, survived: 1 },
    { pclass: 3, sex: 0, age: 14, sibsp: 1, parch: 0, fare: 11.24, survived: 0 },
];

export const sampleDatasets: Dataset[] = [
    {
        id: 'iris',
        name: 'Iris Flowers',
        description: 'Classic dataset for clustering and classification with 4 features and 3 species.',
        rows: irisData.length,
        features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        targetColumn: 'species',
        type: 'clustering',
        data: irisData,
    },
    {
        id: 'customers',
        name: 'Customer Segmentation',
        description: 'Mall customer data with age, income, and spending score for segmentation analysis.',
        rows: customerData.length,
        features: ['age', 'annual_income', 'spending_score'],
        type: 'clustering',
        data: customerData,
    },
    {
        id: 'titanic',
        name: 'Titanic Survival',
        description: 'Simplified Titanic dataset for binary classification (survived/not survived).',
        rows: titanicData.length,
        features: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'],
        targetColumn: 'survived',
        type: 'classification',
        data: titanicData,
    },
];

export const getDatasetById = (id: string): Dataset | undefined => {
    return sampleDatasets.find(d => d.id === id);
};

export const getDatasetInfo = (): DatasetInfo[] => {
    return sampleDatasets.map(({ data, ...info }) => info);
};
