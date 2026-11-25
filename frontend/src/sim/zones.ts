export const PROVENANCE =
  "SYNTHETIC v2 SIMULATION (Seed 20261125 • 8 Bengaluru Zones) — Live interactive telemetry simulation; official VZCrash metrics remain on Overview."

export const ZONES = [
  { name: "Koramangala", latMin: 12.925, latMax: 12.945, lonMin: 77.61, lonMax: 77.64 },
  { name: "Indiranagar", latMin: 12.97, latMax: 12.985, lonMin: 77.635, lonMax: 77.655 },
  { name: "Whitefield", latMin: 12.96, latMax: 12.99, lonMin: 77.72, lonMax: 77.76 },
  { name: "Jayanagar", latMin: 12.92, latMax: 12.94, lonMin: 77.58, lonMax: 77.6 },
  { name: "Malleswaram", latMin: 13.0, latMax: 13.02, lonMin: 77.56, lonMax: 77.58 },
  { name: "Hebbal", latMin: 13.03, latMax: 13.05, lonMin: 77.59, lonMax: 77.61 },
  { name: "Electronic City", latMin: 12.83, latMax: 12.855, lonMin: 77.66, lonMax: 77.69 },
  { name: "Yelahanka", latMin: 13.09, latMax: 13.115, lonMin: 77.58, lonMax: 77.61 },
] as const

const CITY = { latMin: 12.8, latMax: 13.15, lonMin: 77.5, lonMax: 77.8 }

export function zoneCenter(name: string): { lat: number; lng: number } {
  const zone = ZONES.find((item) => item.name === name) ?? ZONES[0]
  return { lat: (zone.latMin + zone.latMax) / 2, lng: (zone.lonMin + zone.lonMax) / 2 }
}

export function nearestZone(lat: number, lng: number): string {
  const ranked = [...ZONES].sort((a, b) => distance(lat, lng, a) - distance(lat, lng, b))
  return ranked[0].name
}

export function insideCity(lat: number, lng: number): boolean {
  return lat >= CITY.latMin && lat <= CITY.latMax && lng >= CITY.lonMin && lng <= CITY.lonMax
}

function distance(lat: number, lng: number, zone: (typeof ZONES)[number]): number {
  const center = zoneCenter(zone.name)
  return (lat - center.lat) ** 2 + (lng - center.lng) ** 2
}
