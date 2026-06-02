import { render, screen } from '@testing-library/react'
import App from './App'
import { describe, it, expect } from 'vitest'

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />)
    expect(screen.getByText(/Intelligent/i)).toBeInTheDocument()
    expect(screen.getByText(/AutoML Platform/i)).toBeInTheDocument()
  })
})
