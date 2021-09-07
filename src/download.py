import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    accesorios_genericos_id = '1Yi4FisqlBxhY_tyK0XTs-WVf5XbdO55t'
    accesorios_genericos_dest = 'Accesorios_Genericos_bd.zip'

    baquerizo_joyeros_id = '1hKgUazK5JHmjBt9Ko7zkkgTiOh17coyL'
    baquerizo_joyeros_dest = 'Baquerizo_Joyeros_bd.zip'

    donasol_id = '1nupwKfcIAmE0NN9rgg6MZ8KsXdwfdGG9'
    donasol_dest = 'Donasol_bd.zip'

    download_file_from_google_drive(
        accesorios_genericos_id, accesorios_genericos_dest)
    download_file_from_google_drive(
        baquerizo_joyeros_id, baquerizo_joyeros_dest)
    download_file_from_google_drive(donasol_id, donasol_dest)
