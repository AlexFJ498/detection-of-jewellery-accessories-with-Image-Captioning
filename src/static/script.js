$(document).ready(function () {
    $('#uploadButton').click(function () {
        if ($('#customFile').val() == '') {
            Swal.fire({
                icon: 'error',
                title: 'Error',
                text: 'Selecciona una imagen.',
                confirmButtonText: 'Ok'
            });
        }

        else {
            if ($('#customFile').val().split('.').pop() == 'jpg' ||
                $('#customFile').val().split('.').pop() == 'jpeg' ||
                $('#customFile').val().split('.').pop() == 'png') {

                if ($('#selectLevel').val() == 0) {
                    Swal.fire({
                        icon: 'error',
                        title: 'Error',
                        text: 'Selecciona un nivel de descripción.',
                        confirmButtonText: 'Ok'
                    });
                }

                else {
                    $('#main_block').fadeOut("slow");

                    $('#main_block').promise().done(function () {
                        var form_data = new FormData($('#form')[0]);

                        $.ajax({
                            type: "POST",
                            url: "/predict",
                            data: form_data,
                            contentType: false,
                            cache: false,
                            processData: false,
                            success: function (data) {
                                image = "../static/" + data.image_path;
                                caption = data.caption;
                                console.log(image);
                                console.log(caption);

                                $('#result_image').attr('src', image);
                                $('#result_caption').html("<h3>" + caption + "</h3>");

                                $('#result_block').fadeIn("slow");
                            }
                        });
                    });
                }
            }

            else {
                Swal.fire({
                    icon: 'error',
                    title: 'Error',
                    text: 'Selecciona un formato compatible (jpg, jpeg, png).',
                    confirmButtonText: 'Ok'
                });
            }
        }
    });

    $('#goBack').click(function () {
        $('#result_block').fadeOut("slow");

        $('#result_block').promise().done(function () {
            $('#result_block').css('display', 'none');
            $('#main_block').fadeIn("slow");
        });
    });
});