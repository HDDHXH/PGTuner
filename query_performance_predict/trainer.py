import os
import torch
from utils import np2ts, save_model, load_model, predict_performance, calculate_errors
from tqdm import tqdm

def dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, save_path, writer, device):
    start_epoch = 0
    best_error = 1000000
    count = 0

    # if os.path.exists(save_path):
    #     model, optimizer, start_epoch = load_model(model, optimizer, save_path)
    #     start_epoch += 1

    for epoch in tqdm(range(start_epoch, args.dipredict_n_epochs),
                      total=len(range(start_epoch, args.dipredict_n_epochs))):
        model.train()

        epoch_loss = 0

        for batch_index, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            batch_feature_train, batch_performance_train = batch_data

            batch_output_train = model(batch_feature_train)

            loss = model.calculate_loss(batch_output_train, batch_performance_train)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        # print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        # writer.add_scalar('Training Epoch Loss', epoch_loss, epoch + 1)

        if epoch < args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 不验证时5个epoch保存一次模型
            save_model(model, optimizer, epoch, save_path)

        if epoch >= args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 开始验证后只有验证效果更好了才保存
            #print('-------------------本轮需要进行验证-------------------')
            model.eval()

            with torch.no_grad():
                #print('------预测性能·------')
                predicted_performances = model(feature_valid)
                predicted_performances_n = performance_scaler.inverse_transform(predicted_performances)
                predicted_performances_n[:, 1:] = torch.pow(10, predicted_performances_n[:, 1:])

                #print('------计算误差------')
                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid, predicted_performances)

                final_mean_error = torch.mean(mean_errors).item()  # 进一步对三个维度取平均
                final_mean_errors_percent = torch.mean(mean_errors_percent).item()
                final_mean_qerror = torch.mean(mean_qerrors).item()

                # print('当前预测误差为：')
                # print(f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(model, optimizer, epoch, save_path)
                    count = 0

                    # print('最佳预测误差为：')
                    # print(
                    #     f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid_raw, predicted_performances_n)
                # print('当前实际预测误差为：')
                # print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')

            if count == args.max_count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break

def Predict_performance_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,  performance_scaler, args, save_path, writer, device):
    start_epoch = 0
    best_error = 1000000
    count = 0

    if os.path.exists(save_path):
        model, optimizer, start_epoch = load_model(model, optimizer, save_path)
        start_epoch += 1

    for epoch in tqdm(range(start_epoch, args.dipredict_n_epochs), total=len(range(start_epoch, args.dipredict_n_epochs))):
        model.train()

        epoch_loss = 0

        for batch_index, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            batch_feature_train, batch_performance_train = batch_data

            _, _, _, batch_output_train = model(batch_feature_train)
            loss = model.calculate_loss(batch_output_train, batch_performance_train)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        # writer.add_scalar('Training Epoch Loss', epoch_loss, epoch + 1)

        if epoch < args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 不验证时5个epoch保存一次模型
            save_model(model, optimizer, epoch, save_path)

        if epoch >= args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 开始验证后只有验证效果更好了才保存
            print('-------------------本轮需要进行验证-------------------')
            model.eval()

            with torch.no_grad():
                print('------预测性能·------')
                _, _, _, predicted_performances = model(feature_valid)
                predicted_performances_n = performance_scaler.inverse_transform(predicted_performances)
                predicted_performances_n[:, 1:] = torch.pow(10, predicted_performances_n[:, 1:])

                print('------计算误差------')
                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid, predicted_performances)

                final_mean_error = torch.mean(mean_errors).item()  # 进一步对三个维度取平均
                final_mean_errors_percent = torch.mean(mean_errors_percent).item()
                final_mean_qerror = torch.mean(mean_qerrors).item()

                print('当前预测误差为：')
                print(f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(model, optimizer, epoch, save_path)
                    count = 0

                    print('最佳预测误差为：')
                    print(
                        f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid_raw, predicted_performances_n)
                print('当前实际预测误差为：')
                print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')

            if count == args.max_count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break

def Predict_loss_train(dataloader, performance_model, loss_model, loss_optimizer, loss_scheduler, feature_valid, real_losses_valid_norm, args, loss_model_save_path, writer, device):
    start_epoch = 0
    best_error = 1000000
    count = 0

    if os.path.exists(loss_model_save_path):
        loss_model, loss_optimizer, start_epoch = load_model(loss_model, loss_optimizer, loss_model_save_path)
        start_epoch += 1

    for epoch in tqdm(range(start_epoch, args.predict_loss_n_epochs), total=len(range(start_epoch, args.predict_loss_n_epochs))):
        loss_model.train()

        epoch_loss = 0

        for batch_index, batch_data in enumerate(dataloader):
            loss_optimizer.zero_grad()

            batch_feature_train, batch_real_losses_train_norm = batch_data

            with torch.no_grad():
                batch_o1, batch_o2, batch_o3, _ = performance_model(batch_feature_train)

            batch_o1 = batch_o1.detach()
            batch_o2 = batch_o2.detach()
            batch_o3 = batch_o3.detach()

            predicted_losses = loss_model(batch_o1, batch_o2, batch_o3)

            loss = loss_model.calculate_loss(predicted_losses, batch_real_losses_train_norm) #loss是预测损失的损失的标量

            epoch_loss += loss.item()

            loss.backward()

            loss_optimizer.step()

        loss_scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

        if epoch < args.predict_loss_valid_epoch and (epoch + 1) % 5 == 0:  # 不验证时5个epoch保存一次模型
            save_model(loss_model, loss_optimizer, epoch, loss_model_save_path)

        if epoch >= args.predict_loss_valid_epoch and (epoch + 1) % 5 == 0:  # 开始验证后只有验证效果更好了才保存
            print('-------------------本轮需要进行验证-------------------')
            loss_model.eval()

            with torch.no_grad():
                print('------预测性能·------')
                batch_o1_valid, batch_o2_valid, batch_o3_valid, _ = performance_model(feature_valid)

                predicted_losses_valid = loss_model(batch_o1_valid, batch_o2_valid, batch_o3_valid)

                print('------计算误差------')
                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(real_losses_valid_norm, predicted_losses_valid)

                final_mean_error = torch.mean(mean_errors).item()  # 进一步对三个维度取平均
                final_mean_errors_percent = torch.mean(mean_errors_percent).item()
                final_mean_qerror = torch.mean(mean_qerrors).item()

                print('当前预测误差为：')
                print(f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(loss_model, loss_optimizer, epoch, loss_model_save_path)
                    count = 0

                    print('最佳预测误差为：')
                    print(
                        f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

            if count == args.max_count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break

def dipredict_train_Bayesian(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, save_path, writer, device):
    start_epoch = 0
    best_error = 1000000
    count = 0

    if os.path.exists(save_path):
        model, optimizer, start_epoch = load_model(model, optimizer, save_path)
        start_epoch += 1

    for epoch in tqdm(range(start_epoch, args.dipredict_n_epochs),
                      total=len(range(start_epoch, args.dipredict_n_epochs))):
        model.train()

        epoch_loss = 0

        for batch_index, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            batch_feature_train, batch_performance_train = batch_data

            loss = model.calculate_loss(batch_feature_train, batch_performance_train, 3, 0)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        # writer.add_scalar('Training Epoch Loss', epoch_loss, epoch + 1)

        if epoch < args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 不验证时5个epoch保存一次模型
            save_model(model, optimizer, epoch, save_path)

        if epoch >= args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 开始验证后只有验证效果更好了才保存
            print('-------------------本轮需要进行验证-------------------')
            model.eval()

            with torch.no_grad():
                print('------预测性能·------')
                predicted_performances = model(feature_valid)
                predicted_performances_n = performance_scaler.inverse_transform(predicted_performances)
                predicted_performances_n[:, 1:] = torch.pow(10, predicted_performances_n[:, 1:])

                print('------计算误差------')
                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid, predicted_performances)

                final_mean_error = torch.mean(mean_errors).item()  # 进一步对三个维度取平均
                final_mean_errors_percent = torch.mean(mean_errors_percent).item()
                final_mean_qerror = torch.mean(mean_qerrors).item()

                print('当前预测误差为：')
                print(f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(model, optimizer, epoch, save_path)
                    count = 0

                    print('最佳预测误差为：')
                    print(
                        f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid_raw, predicted_performances_n)
                print('当前实际预测误差为：')
                print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')

            if count == args.max_count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break


def dipredict_train_with_uncertainty(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, save_path, writer, device):
    start_epoch = 0
    best_error = 1000000
    count = 0

    if os.path.exists(save_path):
        model, optimizer, start_epoch = load_model(model, optimizer, save_path)
        start_epoch += 1

    for epoch in tqdm(range(start_epoch, args.dipredict_n_epochs),
                      total=len(range(start_epoch, args.dipredict_n_epochs))):
        model.train()

        epoch_loss = 0

        for batch_index, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            batch_feature_train, batch_performance_train = batch_data

            batch_mu_train, batch_sigma_train = model(batch_feature_train)

            loss = model.calculate_loss(batch_mu_train, batch_performance_train, batch_sigma_train)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        # writer.add_scalar('Training Epoch Loss', epoch_loss, epoch + 1)

        if epoch < args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 不验证时5个epoch保存一次模型
            save_model(model, optimizer, epoch, save_path)

        if epoch >= args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 开始验证后只有验证效果更好了才保存
            print('-------------------本轮需要进行验证-------------------')
            model.eval()

            with torch.no_grad():
                print('------预测性能·------')
                predicted_performances, predicted_variance = model(feature_valid)
                predicted_performances_n = performance_scaler.inverse_transform(predicted_performances)
                predicted_performances_n[:, 1:] = torch.pow(10, predicted_performances_n[:, 1:])

                print('------计算误差------')
                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid, predicted_performances)

                final_mean_error = torch.mean(mean_errors).item()  # 进一步对三个维度取平均
                final_mean_errors_percent = torch.mean(mean_errors_percent).item()
                final_mean_qerror = torch.mean(mean_qerrors).item()

                print('当前预测误差为：')
                print(f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(model, optimizer, epoch, save_path)
                    count = 0

                    print('最佳预测误差为：')
                    print(
                        f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_valid_raw, predicted_performances_n)
                print('当前实际预测误差为：')
                print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')

                print('预测方差为：')
                print(f'min_variance:{torch.min(predicted_variance, dim=0).values}, mean_variance:{torch.mean(predicted_variance, dim=0)}, max_variance:{torch.max(predicted_variance, dim=0).values}')

            if count == args.max_count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break


def mt_dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                       performance_scaler, args, save_path, writer, device):
    start_epoch = 0
    best_error = 1000000
    count = 0

    if os.path.exists(save_path):
        model, optimizer, start_epoch = load_model(model, optimizer, save_path)
        start_epoch += 1

    for epoch in tqdm(range(start_epoch, args.dipredict_n_epochs),
                      total=len(range(start_epoch, args.dipredict_n_epochs))):
        model.train()

        epoch_loss = 0

        for batch_index, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            batch_feature_train, batch_performance_train = batch_data

            batch_output_train_r, batch_output_train_c, batch_output_train_q = model(batch_feature_train)
            loss = model.calculate_loss(batch_output_train_r, batch_output_train_c, batch_output_train_q,
                                        batch_performance_train)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        writer.add_scalar('Training Epoch Loss', epoch_loss, epoch + 1)

        if epoch < args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 不验证时5个epoch保存一次模型
            save_model(model, optimizer, epoch, save_path)

        if epoch >= args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 开始验证后只有验证效果更好了才保存
            print('-------------------本轮需要进行验证-------------------')
            model.eval()

            with torch.no_grad():
                print('------预测性能·------')
                predicted_performances_r, predicted_performances_c, predicted_performances_q = model(feature_valid)
                predicted_performances = torch.cat(
                    (predicted_performances_r, predicted_performances_c, predicted_performances_q), dim=1)

                predicted_performances_n = predicted_performances.cpu().numpy()
                predicted_performances_n = performance_scaler.inverse_transform(predicted_performances_n)

                print('------计算误差------')
                mean_errors, mean_qerrors = calculate_errors(performance_valid,
                                                             predicted_performances)  # 这里误差是一个3维张量，是所有验证样本的平均
                final_mean_error = torch.mean(mean_errors).item()  # 进一步对三个维度取平均
                final_mean_qerror = torch.mean(mean_qerrors).item()

                print('当前预测误差为：')
                print(f'mean_error:{mean_errors} {final_mean_error}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(model, optimizer, epoch, save_path)
                    count = 0

                    print('最佳预测误差为：')
                    print(
                        f'mean_error:{mean_errors} {final_mean_error}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

                predicted_performances = np2ts(predicted_performances_n).to(device)
                mean_errors, mean_qerrors = calculate_errors(performance_valid_raw, predicted_performances)
                print('当前实际预测误差为：')
                print(f'mean_error:{mean_errors}, mean_qerror:{mean_qerrors}')

            if count == args.max_count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break


def it_dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                       args, save_path, writer, device, flag):
    start_epoch = 0
    best_error = 1000000
    count = 0

    if os.path.exists(save_path):
        model, optimizer, start_epoch = load_model(model, optimizer, save_path)
        start_epoch += 1

    for epoch in tqdm(range(start_epoch, args.dipredict_n_epochs),
                      total=len(range(start_epoch, args.dipredict_n_epochs))):
        model.train()

        epoch_loss = 0

        for batch_index, batch_data in enumerate(dataloader):
            optimizer.zero_grad()

            batch_feature_train, batch_performance_train = batch_data

            batch_output_train = model(batch_feature_train)
            loss = model.calculate_loss(batch_output_train, batch_performance_train)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        # writer.add_scalar('Training Epoch Loss', epoch_loss, epoch + 1)

        if epoch < args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 不验证时5个epoch保存一次模型
            save_model(model, optimizer, epoch, save_path)

        if epoch >= args.dipredict_valid_epoch and (epoch + 1) % 5 == 0:  # 开始验证后只有验证效果更好了才保存
            print('-------------------本轮需要进行验证-------------------')
            model.eval()

            with torch.no_grad():
                print('------预测性能·------')
                predicted_performances = model(feature_valid)

                print('------计算误差------')
                mean_errors, mean_qerrors = calculate_errors(performance_valid, predicted_performances)
                final_mean_error = mean_errors.item()
                final_mean_qerror = mean_qerrors.item()

                print('当前预测误差为：')
                print(f'mean_error:{mean_errors} {final_mean_error}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(model, optimizer, epoch, save_path)
                    count = 0

                    print('最佳预测误差为：')
                    print(
                        f'mean_error:{mean_errors} {final_mean_error}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

                if flag == 'c':
                    predicted_performances = predicted_performances * (torch.tensor(7199.915403).cuda()) + torch.tensor(
                        0.084597).cuda()
                elif flag == 'q':
                    predicted_performances = predicted_performances * (torch.tensor(32335).cuda()) + torch.tensor(
                        1).cuda()

                mean_errors, mean_qerrors = calculate_errors(performance_valid_raw, predicted_performances)
                print('当前实际预测误差为：')
                print(f'mean_error:{mean_errors}, mean_qerror:{mean_qerrors}')

            if count == args.max_count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break

def vae_train(dataloader, model, optimizer, scheduler, feature_valid, feature_valid_raw, feature_scaler, args, save_path, writer, device):
    start_epoch = 0
    best_error = 1000000
    count = 0

    if os.path.exists(save_path):
        model, optimizer, start_epoch = load_model(model, optimizer, save_path)
        start_epoch += 1

    for epoch in tqdm(range(start_epoch, args.vae_epochs), total=len(range(start_epoch, args.vae_epochs))):
        model.train()

        epoch_loss = 0

        for batch_index, batch_data_train in enumerate(dataloader):
            optimizer.zero_grad()

            batch_output_train, batch_mu, batch_logvar = model(batch_data_train)
            loss = model.calculate_loss(batch_data_train, batch_output_train, batch_mu, batch_logvar)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')
        # writer.add_scalar('Training Epoch Loss', epoch_loss, epoch + 1)

        if epoch < args.vae_valid_epoch and (epoch + 1) % 5 == 0:  # 不验证时5个epoch保存一次模型
            save_model(model, optimizer, epoch, save_path)

        if epoch >= args.vae_valid_epoch and (epoch + 1) % 5 == 0:  # 开始验证后只有验证效果更好了才保存
            print('-------------------本轮需要进行验证-------------------')
            model.eval()

            with torch.no_grad():
                print('------预测性能·------')
                predicted_input, _, _ = model(feature_valid)

                predicted_input_n = feature_scaler.inverse_transform(predicted_input)
                predicted_input_n[:, 0] = torch.pow(10, predicted_input_n[:, 0])
                predicted_input_n[:, 2:5] = torch.pow(10, predicted_input_n[:, 2:5])

                print('------计算误差------')
                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(feature_valid, predicted_input)

                final_mean_error = torch.mean(mean_errors).item()  # 进一步对三个维度取平均
                final_mean_errors_percent = torch.mean(mean_errors_percent).item()
                final_mean_qerror = torch.mean(mean_qerrors).item()

                print('当前预测误差为：')
                print(f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')

                if final_mean_error < best_error:
                    best_error = final_mean_error
                    save_model(model, optimizer, epoch, save_path)
                    count = 0

                    print('最佳预测误差为：')
                    print(
                        f'mean_error:{mean_errors} {final_mean_error}, mean_error_percent:{mean_errors_percent} {final_mean_errors_percent}, mean_qerror:{mean_qerrors} {final_mean_qerror}')
                else:
                    count += 1

                mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(feature_valid_raw, predicted_input_n)
                print('当前实际预测误差为：')
                print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')

            if count == args.max_count:  # 模型训练到后面，可能会出现性能下降的情况，这样能尽可能保证模型性能最好
                break