"use client"

import React from "react"

interface ProgressCircleProps {
  value: number
  max: number
  size?: number
  strokeWidth?: number
  className?: string
  showText?: boolean
}

export function ProgressCircle({ 
  value, 
  max, 
  size = 40, 
  strokeWidth = 3,
  className = "",
  showText = false
}: ProgressCircleProps) {
  const percentage = max > 0 ? (value / max) * 100 : 0
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const strokeDashoffset = circumference - (percentage / 100) * circumference
  
  return (
    <div className={`relative inline-flex items-center justify-center ${className}`}>
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-gray-200 dark:text-gray-700"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="text-blue-400 transition-all duration-300"
        />
      </svg>
      {showText && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs font-medium">
            {value}/{max}
          </span>
        </div>
      )}
    </div>
  )
}