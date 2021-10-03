import pgeocode

dist = pgeocode.GeoDistance('us')

print(dist.query_postal_code("19713", "92602"))
