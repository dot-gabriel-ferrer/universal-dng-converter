# RAW Compatibility Guide

Este documento explica las nuevas funcionalidades de compatibilidad RAW añadidas al Universal DNG Converter para resolver errores como `LibRawFileUnsupportedError`.

## Problema Resuelto

Anteriormente, los archivos DNG generados por el convertidor causaban errores en bibliotecas como `rawpy`:

```
rawpy._rawpy.LibRawFileUnsupportedError: b'Unsupported file format or not RAW file'
```

Esto ocurría porque los archivos DNG generados eran técnicamente archivos TIFF con extensión `.dng`, pero no contenían los metadatos y estructura necesarios para ser reconocidos como archivos RAW.

## Nuevas Funcionalidades

### 1. Soporte para Archivos RAW de Entrada

El convertidor ahora puede leer archivos RAW usando `rawpy`:

- **Formatos soportados**: CR2, CR3, NEF, ARW, ORF, RW2, RAF, RAW, DNG
- **Preservación de metadatos**: Información de cámara, niveles de negro/blanco, matriz de color
- **Datos Bayer**: Preserva los datos del sensor sin procesar

### 2. Generación de DNG Compatibles con RAW

#### Modo RAW Habilitado (`raw_output=True`)

```python
from universal_dng_converter import ImageConverter

# Crear convertidor con salida RAW habilitada
converter = ImageConverter(raw_output=True)

# Convertir a DNG compatible con RAW
result = converter.convert_to_raw_compatible_dng(
    input_path="imagen.jpg",
    output_dir="./output/",
    validate_raw=True  # Valida con rawpy si está disponible
)
```

#### Características de los DNG RAW:

- **Metadatos DNG**: Incluye tags específicos como `DNGVersion`, `WhiteLevel`, `BlackLevel`
- **Compresión LZW**: Mejor compatibilidad que ZIP para RAW
- **Estructura en mosaico**: Organización tiled para mejor rendimiento
- **Interpretación fotométrica CFA**: Para datos de sensor

### 3. Manejo de Errores RAW

#### Función de Reparación Automática

```python
# Reparar un DNG problemático
fixed_file = converter.handle_raw_processing_error(
    dng_path="archivo_problematico.dng",
    output_dir="./fixed/",
    fallback_format="tiff"  # Formato alternativo si falla la reparación
)
```

### 4. CLI Mejorado

#### Nuevas opciones de línea de comandos:

```bash
# Crear DNG compatible con RAW
universal-dng-converter --input imagen.jpg --output ./output/ --raw-compatible

# Validar compatibilidad RAW
universal-dng-converter --input imagen.jpg --output ./output/ --raw-compatible --validate-raw

# Reparar DNG problemático
universal-dng-converter --fix-raw-errors problematico.dng --output ./fixed/ --fallback-format tiff
```

## Instalación de Dependencias

Para usar las funcionalidades RAW completas, instala `rawpy`:

```bash
pip install rawpy>=0.17.0
```

**Nota**: `rawpy` es opcional. Si no está instalado, el convertidor:
- Usará cargadores alternativos para archivos DNG
- No podrá validar compatibilidad RAW
- Seguirá funcionando para otros formatos

## Ejemplos de Uso

### Ejemplo 1: Conversión Básica RAW

```python
from universal_dng_converter import ImageConverter

converter = ImageConverter(raw_output=True)

# Convertir archivo RAW de cámara
result = converter.convert_image("IMG_1234.CR2", "output.dng")

# Verificar que rawpy puede leerlo
import rawpy
with rawpy.imread("output.dng") as raw:
    print(f"Éxito: {raw.raw_image.shape}")
```

### Ejemplo 2: Procesamiento por Lotes

```python
# Convertir todos los archivos RAW de un directorio
results = converter.batch_convert(
    input_dir="./raw_photos/",
    output_dir="./dng_output/",
    recursive=True
)

# Ver estadísticas
print(f"Convertidos: {converter.stats['converted']}")
print(f"Errores: {converter.stats['errors']}")
```

### Ejemplo 3: Manejo de Errores

```python
try:
    # Intentar leer DNG con rawpy
    with rawpy.imread("archivo.dng") as raw:
        data = raw.raw_image
except rawpy.LibRawFileUnsupportedError:
    # Reparar automáticamente
    print("Error RAW detectado, reparando...")
    fixed = converter.handle_raw_processing_error(
        "archivo.dng",
        "./fixed/"
    )
    if fixed:
        print(f"Reparado: {fixed}")
```

## Diferencias entre Modos

| Característica | Modo Estándar | Modo RAW |
|----------------|---------------|----------|
| Extensión | `.dng` | `.dng` |
| Compresión | ZIP/LZW | LZW |
| Metadatos DNG | Básicos | Completos |
| Compatibilidad rawpy | Limitada | Completa |
| Tamaño archivo | Menor | Ligeramente mayor |
| Estructura | TIFF simple | TIFF con estructura RAW |

## Validación de Compatibilidad

El convertidor incluye validación automática:

```python
# La validación se ejecuta automáticamente
result = converter.convert_to_raw_compatible_dng(
    "input.jpg",
    "./output/",
    validate_raw=True
)

# O validar manualmente
is_compatible = converter._validate_raw_compatibility("output.dng")
```

## Resolución de Problemas

### Error: "rawpy not available"
- Instalar: `pip install rawpy`
- O usar sin validación: `validate_raw=False`

### DNG creado pero rawpy falla
- Verificar que el archivo fuente tiene datos válidos
- Intentar con diferentes configuraciones de `bit_depth` y `scaling_method`
- Usar `handle_raw_processing_error` para generar versión alternativa

### Archivos muy grandes
- Ajustar `quality` (menor valor = menor tamaño)
- Usar `bit_depth=8` para imágenes que no requieren 16 bits
- Considerar formato alternativo con `fallback_format`

## Configuración Recomendada

Para máxima compatibilidad RAW:

```python
converter = ImageConverter(
    raw_output=True,
    bit_depth=16,           # Preservar precision
    scaling_method="none",  # Preservar valores originales
    quality=95             # Balance tamaño/calidad
)
```

Para compatibilidad general:

```python
converter = ImageConverter(
    raw_output=False,
    bit_depth=16,
    scaling_method="auto",
    quality=95
)
```
