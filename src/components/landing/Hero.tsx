"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Play, Database, Cpu, BarChart3 } from "lucide-react";

export const Hero = () => {
    return (
        <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 overflow-hidden bg-gradient-to-b from-gray-50 to-white">
            {/* Background elements */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full -z-10">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-[#004040]/5 blur-[120px]" />
                <div className="absolute bottom-[10%] right-[-10%] w-[30%] h-[30%] rounded-full bg-[#004040]/10 blur-[120px]" />
                <div className="absolute inset-0 bg-[linear-gradient(to_right,#00404008_1px,transparent_1px),linear-gradient(to_bottom,#00404008_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />
            </div>

            <div className="container max-w-6xl mx-auto px-4 md:px-6">
                <div className="flex flex-col items-center text-center">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <Badge variant="outline" className="mb-6 border-[#004040]/30 bg-[#004040]/5 text-[#004040] px-4 py-1">
                            100% Open Source & Free
                        </Badge>
                    </motion.div>

                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="text-4xl md:text-7xl font-bold tracking-tight text-gray-900 mb-6"
                    >
                        Run <span className="text-[#004040]">K-Means, Naive Bayes</span>, <br /> and more in the browser.
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        className="text-lg md:text-xl text-gray-600 max-w-2xl mb-10 leading-relaxed"
                    >
                        Upload CSV, tune parameters, and interpret results with clear metrics and explanations. No setup, no Python, just data.
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.3 }}
                        className="flex flex-col sm:flex-row gap-4"
                    >
                        <Link href="#demo">
                            <Button size="lg" className="bg-[#004040] hover:bg-[#003333] text-white min-w-[160px] h-12 text-base">
                                Try Demo <Play className="ml-2 w-4 h-4 fill-current" />
                            </Button>
                        </Link>
                        <Link href="#algorithms">
                            <Button size="lg" variant="outline" className="border-gray-300 text-gray-700 hover:bg-gray-100 min-w-[160px] h-12 text-base">
                                Browse Algorithms
                            </Button>
                        </Link>
                    </motion.div>

                    {/* Floating Cards Preview */}
                    <div className="mt-20 relative w-full max-w-4xl mx-auto h-[300px] md:h-[400px]">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.8, rotateX: 20 }}
                            animate={{ opacity: 1, scale: 1, rotateX: 0 }}
                            transition={{ duration: 0.8, delay: 0.5 }}
                            className="absolute inset-0 bg-white border border-gray-200 rounded-2xl backdrop-blur-sm overflow-hidden shadow-2xl"
                        >
                            <div className="flex border-b border-gray-200 p-4 items-center gap-2 bg-gray-50">
                                <div className="flex gap-1.5">
                                    <div className="w-3 h-3 rounded-full bg-rose-400" />
                                    <div className="w-3 h-3 rounded-full bg-amber-400" />
                                    <div className="w-3 h-3 rounded-full bg-emerald-400" />
                                </div>
                                <div className="mx-auto text-xs text-gray-500 font-mono">kmeans_analysis.lab</div>
                            </div>
                            <div className="p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
                                {/* Upload Data Card */}
                                <div className="space-y-3">
                                    <div className="flex items-center gap-2">
                                        <Database className="w-4 h-4 text-[#004040]" />
                                        <span className="text-sm font-semibold text-gray-800">Upload Data</span>
                                    </div>
                                    <div className="h-32 w-full bg-gradient-to-br from-[#004040]/5 to-[#004040]/10 rounded-lg border-2 border-dashed border-[#004040]/20 flex flex-col items-center justify-center gap-2 cursor-pointer hover:border-[#004040]/40 transition-colors">
                                        <Database className="w-8 h-8 text-[#004040]/60" />
                                        <span className="text-xs text-gray-500">customers.csv</span>
                                        <span className="text-[10px] text-emerald-600 font-medium">✓ 1,247 rows loaded</span>
                                    </div>
                                </div>

                                {/* Configure Algorithm Card */}
                                <div className="space-y-3">
                                    <div className="flex items-center gap-2">
                                        <Cpu className="w-4 h-4 text-violet-600" />
                                        <span className="text-sm font-semibold text-gray-800">Configure</span>
                                    </div>
                                    <div className="h-32 w-full bg-white rounded-lg border border-gray-200 p-4 space-y-3">
                                        <div className="flex justify-between items-center">
                                            <span className="text-xs text-gray-500">Algorithm</span>
                                            <span className="text-xs font-medium text-[#004040] bg-[#004040]/10 px-2 py-0.5 rounded">K-Means</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-xs text-gray-500">Clusters (K)</span>
                                            <span className="text-xs font-mono font-bold text-gray-800">5</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-xs text-gray-500">Iterations</span>
                                            <span className="text-xs font-mono font-bold text-gray-800">100</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Results Card */}
                                <div className="space-y-3">
                                    <div className="flex items-center gap-2">
                                        <BarChart3 className="w-4 h-4 text-emerald-600" />
                                        <span className="text-sm font-semibold text-gray-800">Results</span>
                                    </div>
                                    <div className="h-32 w-full bg-white rounded-lg border border-gray-200 p-4 space-y-3">
                                        <div className="flex justify-between items-center">
                                            <span className="text-xs text-gray-500">Inertia</span>
                                            <span className="text-xs font-mono font-bold text-emerald-600">234.56</span>
                                        </div>
                                        <div className="flex justify-between items-center">
                                            <span className="text-xs text-gray-500">Silhouette</span>
                                            <span className="text-xs font-mono font-bold text-emerald-600">0.82</span>
                                        </div>
                                        <div className="w-full bg-gray-100 rounded-full h-2 mt-2">
                                            <div className="bg-[#004040] h-2 rounded-full" style={{ width: '82%' }}></div>
                                        </div>
                                        <div className="text-[10px] text-center text-gray-500">Cluster Quality: Excellent</div>
                                    </div>
                                </div>
                            </div>
                        </motion.div>

                        {/* Accent floating elements */}
                        <motion.div
                            animate={{ y: [0, -10, 0] }}
                            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                            className="absolute -top-6 -right-6 md:-top-10 md:-right-10 bg-white border border-gray-200 p-4 rounded-xl shadow-xl z-20 hidden md:block"
                        >
                            <div className="flex items-center gap-3">
                                <div className="bg-emerald-500/10 p-2 rounded-lg">
                                    <BarChart3 className="w-5 h-5 text-emerald-600" />
                                </div>
                                <div>
                                    <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold">Accuracy</div>
                                    <div className="text-lg font-bold text-gray-900">98.2%</div>
                                </div>
                            </div>
                        </motion.div>

                        <motion.div
                            animate={{ y: [0, 10, 0] }}
                            transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                            className="absolute -bottom-6 -left-6 md:-bottom-10 md:-left-10 bg-white border border-gray-200 p-4 rounded-xl shadow-xl z-20 hidden md:block"
                        >
                            <div className="flex items-center gap-3">
                                <div className="bg-[#004040]/10 p-2 rounded-lg">
                                    <Cpu className="w-5 h-5 text-[#004040]" />
                                </div>
                                <div>
                                    <div className="text-[10px] text-gray-500 uppercase tracking-wider font-bold">Clusters</div>
                                    <div className="text-lg font-bold text-gray-900">K = 5</div>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                </div>
            </div>
        </section>
    );
};
