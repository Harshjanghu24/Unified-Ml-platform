import { http, HttpResponse } from 'msw'

export const handlers = [
  http.get('http://localhost:8000/', () => {
    return HttpResponse.json({
      name: "Unified Supervised Learning Platform",
      version: "1.0.0",
      status: "running",
      docs: "/docs",
    })
  }),
]
