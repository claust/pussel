param name string
param location string

resource acr 'Microsoft.ContainerRegistry/registries@2021-06-01-preview' = {
  name: name
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

output name string = acr.name
output loginServer string = acr.properties.loginServer
