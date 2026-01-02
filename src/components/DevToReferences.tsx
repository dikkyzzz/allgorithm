"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ExternalLink, BookOpen, Loader2 } from "lucide-react";

interface DevToArticle {
    id: number;
    title: string;
    url: string;
    user: {
        name: string;
        username: string;
    };
    reading_time_minutes: number;
    positive_reactions_count: number;
    published_at: string;
}

interface DevToReferencesProps {
    algorithmId: string;
    algorithmName: string;
}

// Map algorithm IDs to Dev.to search tags
const algorithmTags: Record<string, string[]> = {
    'k-means': ['kmeans', 'clustering', 'machinelearning'],
    'naive-bayes': ['naivebayes', 'classification', 'machinelearning'],
    'logistic-regression': ['logisticregression', 'classification', 'machinelearning'],
    'linear-regression': ['linearregression', 'regression', 'machinelearning'],
    'pca': ['pca', 'dimensionalityreduction', 'machinelearning'],
    'tf-idf-nb': ['tfidf', 'nlp', 'textclassification'],
    'dbscan': ['dbscan', 'clustering', 'machinelearning'],
    'svm': ['svm', 'classification', 'machinelearning'],
};

export const DevToReferences = ({ algorithmId, algorithmName }: DevToReferencesProps) => {
    const [articles, setArticles] = useState<DevToArticle[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchArticles = async () => {
            setLoading(true);
            setError(null);

            try {
                const tags = algorithmTags[algorithmId] || ['machinelearning'];
                // Try main tag first
                let response = await fetch(
                    `https://dev.to/api/articles?tag=${tags[0]}&per_page=5`
                );

                if (!response.ok) {
                    throw new Error('Failed to fetch');
                }

                let data: DevToArticle[] = await response.json();

                // If no results, try fallback tag
                if (data.length === 0 && tags[1]) {
                    response = await fetch(
                        `https://dev.to/api/articles?tag=${tags[1]}&per_page=5`
                    );
                    data = await response.json();
                }

                // Further fallback to general ML
                if (data.length === 0) {
                    response = await fetch(
                        `https://dev.to/api/articles?tag=machinelearning&per_page=5`
                    );
                    data = await response.json();
                }

                setArticles(data.slice(0, 4)); // Limit to 4 articles
            } catch (err) {
                setError('Could not load articles');
                console.error('Dev.to API error:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchArticles();
    }, [algorithmId]);

    if (loading) {
        return (
            <Card className="bg-white border-gray-200 shadow-sm">
                <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base text-gray-900">
                        <BookOpen className="w-4 h-4" />
                        Learn More
                    </CardTitle>
                </CardHeader>
                <CardContent className="flex items-center justify-center py-8">
                    <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
                </CardContent>
            </Card>
        );
    }

    if (error || articles.length === 0) {
        return (
            <Card className="bg-white border-gray-200 shadow-sm">
                <CardHeader className="pb-3">
                    <CardTitle className="flex items-center gap-2 text-base text-gray-900">
                        <BookOpen className="w-4 h-4" />
                        Learn More
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-gray-500 text-center py-4">
                        Search for &quot;{algorithmName}&quot; on Dev.to for related articles.
                    </p>
                    <Link
                        href={`https://dev.to/search?q=${encodeURIComponent(algorithmName)}`}
                        target="_blank"
                        className="flex items-center justify-center gap-2 text-sm text-[#004040] hover:underline"
                    >
                        <ExternalLink className="w-4 h-4" />
                        Browse on Dev.to
                    </Link>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="bg-white border-gray-200 shadow-sm">
            <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base text-gray-900">
                    <BookOpen className="w-4 h-4" />
                    Learn More
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
                {articles.map((article) => (
                    <Link
                        key={article.id}
                        href={article.url}
                        target="_blank"
                        className="block p-3 rounded-lg hover:bg-gray-50 border border-transparent hover:border-gray-200 transition-all group"
                    >
                        <h4 className="text-sm font-medium text-gray-900 group-hover:text-[#004040] line-clamp-2 mb-1">
                            {article.title}
                        </h4>
                        <div className="flex items-center gap-3 text-xs text-gray-500">
                            <span>{article.user.name}</span>
                            <span>•</span>
                            <span>{article.reading_time_minutes} min read</span>
                            <span>•</span>
                            <span>❤️ {article.positive_reactions_count}</span>
                        </div>
                    </Link>
                ))}
                <Link
                    href={`https://dev.to/search?q=${encodeURIComponent(algorithmName)}`}
                    target="_blank"
                    className="flex items-center justify-center gap-2 pt-2 text-sm text-[#004040] hover:underline"
                >
                    <ExternalLink className="w-4 h-4" />
                    More on Dev.to
                </Link>
            </CardContent>
        </Card>
    );
};
