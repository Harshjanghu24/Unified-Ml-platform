import { describe, it, expect } from 'vitest'
import axios from 'axios'

describe('MSW Mocking', () => {
  it('mocks the root endpoint', async () => {
    const response = await axios.get('http://localhost:8000/')
    expect(response.status).toBe(200)
    expect(response.data.name).toBe('Unified Supervised Learning Platform')
    expect(response.data.status).toBe('running')
  })
})
